//! OpenAI Chat Completions streaming (SSE) implementation.
//!
//! Provides helper types and functions for streaming responses from
//! OpenAI-compatible APIs (`POST /chat/completions` with `"stream": true`).
//! Used by [`RigAdapter`](super::rig_adapter::RigAdapter) when
//! [`StreamingConfig`] is configured.

use std::time::Duration;

use eventsource_stream::Eventsource;
use futures::StreamExt;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{json, Value};

use crate::llm::error::LlmError;
use crate::llm::provider::{
    ChatMessage, FinishReason, ToolCall, ToolCompletionRequest, ToolCompletionResponse,
    ToolDefinition,
};

/// Configuration for direct HTTP streaming, bypassing rig-core.
pub(crate) struct StreamingConfig {
    pub client: reqwest::Client,
    pub base_url: String,
    pub api_key: SecretString,
}

/// Send a streaming Chat Completions request and forward text chunks.
///
/// - Text delta chunks are sent to `chunk_tx` as they arrive.
/// - Tool call arguments are accumulated across chunks.
/// - The final [`ToolCompletionResponse`] is returned with complete content.
///
/// Timeout: 30 s idle timeout per SSE event (same as codex_chatgpt).
pub(crate) async fn stream_chat_completion(
    config: &StreamingConfig,
    request: &ToolCompletionRequest,
    model: &str,
    strict_tools_schema: bool,
    chunk_tx: tokio::sync::mpsc::UnboundedSender<String>,
) -> Result<ToolCompletionResponse, LlmError> {
    let body = build_request_body(request, model, strict_tools_schema);
    let url = format!(
        "{}/chat/completions",
        config.base_url.trim_end_matches('/')
    );

    let resp = config
        .client
        .post(&url)
        .bearer_auth(config.api_key.expose_secret())
        .header("Content-Type", "application/json")
        .header("Accept", "text/event-stream")
        .json(&body)
        .timeout(Duration::from_secs(300))
        .send()
        .await
        .map_err(|e| LlmError::RequestFailed {
            provider: "openai_streaming".into(),
            reason: format!("HTTP request failed: {e}"),
        })?;

    let status = resp.status();
    if !status.is_success() {
        let body_text = resp.text().await.unwrap_or_default();
        return Err(map_error_response(status.as_u16(), &body_text));
    }

    let stream = resp
        .bytes_stream()
        .map(|chunk| chunk.map_err(|e| e.to_string()));

    parse_sse_stream(stream, chunk_tx).await
}

// ── Request body construction ───────────────────────────────────────────

fn build_request_body(
    request: &ToolCompletionRequest,
    model: &str,
    strict_tools_schema: bool,
) -> Value {
    let model_name = request.model.as_deref().unwrap_or(model);
    let messages = messages_to_json(&request.messages);
    let tools = tools_to_json(&request.tools, strict_tools_schema);

    let mut body = json!({
        "model": model_name,
        "messages": messages,
        "stream": true,
        "stream_options": { "include_usage": true },
    });

    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
        if let Some(ref choice) = request.tool_choice {
            body["tool_choice"] = json!(choice);
        }
    }
    if let Some(temp) = request.temperature {
        body["temperature"] = json!(round_f32(temp));
    }
    if let Some(max) = request.max_tokens {
        body["max_tokens"] = json!(max);
    }

    body
}

fn messages_to_json(messages: &[ChatMessage]) -> Vec<Value> {
    messages.iter().filter_map(message_to_json).collect()
}

fn message_to_json(msg: &ChatMessage) -> Option<Value> {
    use crate::llm::provider::Role;
    match msg.role {
        Role::System => Some(json!({"role": "system", "content": &msg.content})),
        Role::User => {
            if msg.content.is_empty() && msg.content_parts.is_empty() {
                return None;
            }
            if msg.content_parts.is_empty() {
                Some(json!({"role": "user", "content": &msg.content}))
            } else {
                let parts = multimodal_parts(msg);
                Some(json!({"role": "user", "content": parts}))
            }
        }
        Role::Assistant => {
            let mut obj = json!({"role": "assistant"});
            if !msg.content.is_empty() {
                obj["content"] = json!(&msg.content);
            }
            if let Some(ref tcs) = msg.tool_calls {
                let calls: Vec<Value> = tcs
                    .iter()
                    .map(|tc| {
                        json!({
                            "id": &tc.id,
                            "type": "function",
                            "function": {
                                "name": &tc.name,
                                "arguments": tc.arguments.to_string(),
                            }
                        })
                    })
                    .collect();
                obj["tool_calls"] = Value::Array(calls);
            }
            Some(obj)
        }
        Role::Tool => Some(json!({
            "role": "tool",
            "tool_call_id": msg.tool_call_id.as_deref().unwrap_or(""),
            "content": &msg.content,
        })),
    }
}

fn multimodal_parts(msg: &ChatMessage) -> Vec<Value> {
    use crate::llm::provider::ContentPart;
    let mut parts = vec![json!({"type": "text", "text": &msg.content})];
    for part in &msg.content_parts {
        if let ContentPart::ImageUrl { image_url } = part {
            parts.push(json!({
                "type": "image_url",
                "image_url": { "url": &image_url.url }
            }));
        }
    }
    parts
}

fn tools_to_json(tools: &[ToolDefinition], strict_schema: bool) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            let params = if strict_schema {
                super::rig_adapter::normalize_schema_strict(&t.parameters)
            } else {
                t.parameters.clone()
            };
            json!({
                "type": "function",
                "function": {
                    "name": &t.name,
                    "description": &t.description,
                    "parameters": params,
                }
            })
        })
        .collect()
}

fn round_f32(val: f32) -> f64 {
    ((val as f64) * 1_000_000.0).round() / 1_000_000.0
}

// ── SSE stream parsing ──────────────────────────────────────────────────

/// In-progress tool call accumulator.
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Accumulated state from SSE events.
struct StreamState {
    text: String,
    reasoning: String,
    refusal: String,
    tool_calls: Vec<PendingToolCall>,
    finish_reason: FinishReason,
    input_tokens: u32,
    output_tokens: u32,
}

async fn parse_sse_stream<S>(
    stream: S,
    chunk_tx: tokio::sync::mpsc::UnboundedSender<String>,
) -> Result<ToolCompletionResponse, LlmError>
where
    S: futures::Stream<Item = Result<bytes::Bytes, String>> + Unpin,
{
    let idle_timeout = Duration::from_secs(30);
    let mut sse = stream.eventsource();
    let mut state = StreamState {
        text: String::new(),
        reasoning: String::new(),
        refusal: String::new(),
        tool_calls: Vec::new(),
        finish_reason: FinishReason::Stop,
        input_tokens: 0,
        output_tokens: 0,
    };

    loop {
        match tokio::time::timeout(idle_timeout, sse.next()).await {
            Ok(Some(Ok(event))) => {
                // Handle SSE error events from some compatible APIs.
                if event.event == "error" {
                    return Err(LlmError::RequestFailed {
                        provider: "openai_streaming".into(),
                        reason: format!("SSE error event: {}", event.data),
                    });
                }
                let data = event.data.trim().to_string();
                if data.is_empty() {
                    continue;
                }
                if data == "[DONE]" {
                    break;
                }
                let parsed: Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                handle_chunk(&parsed, &chunk_tx, &mut state);
            }
            Ok(Some(Err(e))) => {
                return Err(LlmError::RequestFailed {
                    provider: "openai_streaming".into(),
                    reason: format!("SSE stream error: {e}"),
                });
            }
            Ok(None) => break,
            Err(_) => {
                return Err(LlmError::RequestFailed {
                    provider: "openai_streaming".into(),
                    reason: format!(
                        "Timed out waiting for SSE event after {}s",
                        idle_timeout.as_secs()
                    ),
                });
            }
        }
    }

    // Build reasoning string (from o1/o3 reasoning models).
    let reasoning_opt = if state.reasoning.is_empty() {
        None
    } else {
        Some(state.reasoning)
    };

    let tool_calls: Vec<ToolCall> = state
        .tool_calls
        .into_iter()
        .filter_map(|tc| {
            let args: Value = serde_json::from_str(&tc.arguments).unwrap_or(Value::Object(
                serde_json::Map::new(),
            ));
            if tc.name.is_empty() {
                return None;
            }
            Some(ToolCall {
                id: tc.id,
                name: tc.name,
                arguments: args,
                reasoning: reasoning_opt.clone(),
            })
        })
        .collect();

    // If the model refused, surface it as the content.
    let content = if !state.refusal.is_empty() {
        Some(format!("[Model refused] {}", state.refusal))
    } else if state.text.is_empty() {
        None
    } else {
        Some(state.text)
    };

    Ok(ToolCompletionResponse {
        content,
        tool_calls,
        input_tokens: state.input_tokens,
        output_tokens: state.output_tokens,
        finish_reason: state.finish_reason,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
    })
}

fn handle_chunk(
    parsed: &Value,
    chunk_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    state: &mut StreamState,
) {
    // Extract usage from the final chunk (stream_options.include_usage).
    if let Some(usage) = parsed.get("usage") {
        state.input_tokens = usage
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        state.output_tokens = usage
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
    }

    let choice = match parsed.get("choices").and_then(|c| c.get(0)) {
        Some(c) => c,
        None => return,
    };

    // Parse finish_reason.
    if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
        state.finish_reason = match reason {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolUse,
            "content_filter" => FinishReason::ContentFilter,
            _ => FinishReason::Unknown,
        };
    }

    let delta = match choice.get("delta") {
        Some(d) => d,
        None => return,
    };

    // Text content delta.
    if let Some(content) = delta.get("content").and_then(Value::as_str) {
        if !content.is_empty() {
            state.text.push_str(content);
            let _ = chunk_tx.send(content.to_string());
        }
    }

    // Reasoning delta (o1/o3 models).
    if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str) {
        state.reasoning.push_str(reasoning);
    }
    // Some providers use "reasoning" instead of "reasoning_content".
    if let Some(reasoning) = delta.get("reasoning").and_then(Value::as_str) {
        if state.reasoning.is_empty() || !delta.get("reasoning_content").is_some() {
            state.reasoning.push_str(reasoning);
        }
    }

    // Refusal delta (model refused to answer).
    if let Some(refusal) = delta.get("refusal").and_then(Value::as_str) {
        state.refusal.push_str(refusal);
    }

    // Tool call deltas.
    if let Some(Value::Array(tcs)) = delta.get("tool_calls") {
        for tc_delta in tcs {
            let idx = tc_delta
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;

            // Ensure the tool_calls vec is large enough.
            while state.tool_calls.len() <= idx {
                state.tool_calls.push(PendingToolCall {
                    id: String::new(),
                    name: String::new(),
                    arguments: String::new(),
                });
            }

            let pending = &mut state.tool_calls[idx];

            if let Some(id) = tc_delta.get("id").and_then(Value::as_str) {
                pending.id = id.to_string();
            }
            if let Some(func) = tc_delta.get("function") {
                if let Some(name) = func.get("name").and_then(Value::as_str) {
                    pending.name = name.to_string();
                }
                if let Some(args) = func.get("arguments").and_then(Value::as_str) {
                    pending.arguments.push_str(args);
                }
            }
        }
    }
}

// ── Error mapping ───────────────────────────────────────────────────────

fn map_error_response(status: u16, body: &str) -> LlmError {
    // Check for context-length-exceeded in the error body.
    if let Ok(parsed) = serde_json::from_str::<Value>(body) {
        if let Some(err_msg) = parsed
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
        {
            let lower = err_msg.to_lowercase();
            if lower.contains("context length")
                || lower.contains("maximum context")
                || lower.contains("token limit")
            {
                return LlmError::ContextLengthExceeded {
                    used: 0,
                    limit: 0,
                };
            }
        }
    }

    LlmError::RequestFailed {
        provider: "openai_streaming".into(),
        reason: format!("HTTP {status}: {}", truncate_body(body)),
    }
}

fn truncate_body(body: &str) -> &str {
    if body.len() > 500 {
        &body[..500]
    } else {
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_request_body_basic() {
        let request = ToolCompletionRequest::new(
            vec![
                ChatMessage::system("You are helpful."),
                ChatMessage::user("Hello"),
            ],
            vec![],
        )
        .with_temperature(0.7)
        .with_max_tokens(4096);

        let body = build_request_body(&request, "gpt-4o", true);

        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"].as_array().map(|a| a.len()), Some(2));
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn build_request_body_with_tools() {
        let request = ToolCompletionRequest::new(
            vec![ChatMessage::user("search")],
            vec![ToolDefinition {
                name: "web_search".into(),
                description: "Search the web".into(),
                parameters: serde_json::json!({"type": "object"}),
            }],
        )
        .with_tool_choice("auto");

        let body = build_request_body(&request, "gpt-4o", false);

        assert!(body["tools"].as_array().is_some_and(|a| a.len() == 1));
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn message_to_json_all_roles() {
        let sys = message_to_json(&ChatMessage::system("sys")).expect("system");
        assert_eq!(sys["role"], "system");

        let user = message_to_json(&ChatMessage::user("hi")).expect("user");
        assert_eq!(user["role"], "user");

        let asst = message_to_json(&ChatMessage::assistant("ok")).expect("assistant");
        assert_eq!(asst["role"], "assistant");

        let tool =
            message_to_json(&ChatMessage::tool_result("tc1", "search", "result")).expect("tool");
        assert_eq!(tool["role"], "tool");
        assert_eq!(tool["tool_call_id"], "tc1");
    }

    #[test]
    fn message_to_json_assistant_with_tool_calls() {
        let msg = ChatMessage::assistant_with_tool_calls(
            Some("I'll search.".into()),
            vec![ToolCall {
                id: "tc_1".into(),
                name: "web_search".into(),
                arguments: serde_json::json!({"q": "rust"}),
                reasoning: None,
            }],
        );
        let json = message_to_json(&msg).expect("should produce json");
        assert_eq!(json["content"], "I'll search.");
        assert!(json["tool_calls"].as_array().is_some_and(|a| a.len() == 1));
        assert_eq!(json["tool_calls"][0]["function"]["name"], "web_search");
        // Arguments must be a JSON string in the wire format.
        assert!(json["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .is_some());
    }

    #[test]
    fn empty_user_message_skipped() {
        assert!(message_to_json(&ChatMessage::user("")).is_none());
    }

    #[tokio::test]
    async fn parse_sse_text_only() {
        let events = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n",
        );
        let stream =
            futures::stream::iter(vec![Ok::<bytes::Bytes, String>(bytes::Bytes::from(events))]);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        let result = parse_sse_stream(stream, tx).await.expect("should parse");

        assert_eq!(result.content.as_deref(), Some("Hello world"));
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.finish_reason, FinishReason::Stop);
        assert_eq!(result.input_tokens, 10);
        assert_eq!(result.output_tokens, 2);

        // Chunks should have been sent.
        let mut chunks = Vec::new();
        while let Ok(c) = rx.try_recv() {
            chunks.push(c);
        }
        assert_eq!(chunks, vec!["Hello", " world"]);
    }

    #[tokio::test]
    async fn parse_sse_tool_calls() {
        let events = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"\"}}]},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"q\\\":\"}}]},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"rust\\\"}\"}}]},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let stream =
            futures::stream::iter(vec![Ok::<bytes::Bytes, String>(bytes::Bytes::from(events))]);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

        let result = parse_sse_stream(stream, tx).await.expect("should parse");

        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
        assert_eq!(result.tool_calls[0].arguments, serde_json::json!({"q": "rust"}));
        assert_eq!(result.finish_reason, FinishReason::ToolUse);
    }

    #[test]
    fn map_error_context_length() {
        let body = r#"{"error":{"message":"This model's maximum context length is 8192 tokens"}}"#;
        let err = map_error_response(400, body);
        assert!(matches!(err, LlmError::ContextLengthExceeded { .. }));
    }

    #[tokio::test]
    async fn parse_sse_reasoning_delta() {
        let events = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Let me think\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"... about this\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"The answer is 42.\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let stream =
            futures::stream::iter(vec![Ok::<bytes::Bytes, String>(bytes::Bytes::from(events))]);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

        let result = parse_sse_stream(stream, tx).await.expect("should parse");

        assert_eq!(result.content.as_deref(), Some("The answer is 42."));
        // Reasoning is not surfaced in content but stored for tool calls.
    }

    #[tokio::test]
    async fn parse_sse_refusal() {
        let events = concat!(
            "data: {\"choices\":[{\"delta\":{\"refusal\":\"I cannot help with that.\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let stream =
            futures::stream::iter(vec![Ok::<bytes::Bytes, String>(bytes::Bytes::from(events))]);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

        let result = parse_sse_stream(stream, tx).await.expect("should parse");

        assert!(result
            .content
            .as_deref()
            .is_some_and(|c| c.contains("Model refused")));
        assert!(result
            .content
            .as_deref()
            .is_some_and(|c| c.contains("I cannot help with that.")));
    }

    #[tokio::test]
    async fn parse_sse_content_filter() {
        let events = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"content_filter\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let stream =
            futures::stream::iter(vec![Ok::<bytes::Bytes, String>(bytes::Bytes::from(events))]);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

        let result = parse_sse_stream(stream, tx).await.expect("should parse");

        assert_eq!(result.finish_reason, FinishReason::ContentFilter);
        // Partial content is still returned for caller to decide what to do.
        assert_eq!(result.content.as_deref(), Some("partial"));
    }

    #[tokio::test]
    async fn parse_sse_error_event() {
        // Some APIs send `event: error` instead of embedding errors in data.
        let events = "event: error\ndata: {\"message\":\"rate limited\"}\n\n";
        let stream =
            futures::stream::iter(vec![Ok::<bytes::Bytes, String>(bytes::Bytes::from(events))]);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

        let result = parse_sse_stream(stream, tx).await;

        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("SSE error event"));
    }

    #[tokio::test]
    async fn parse_sse_tool_calls_with_reasoning() {
        let events = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"I should search\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"tc1\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\\\"test\\\"}\"}}]},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let stream =
            futures::stream::iter(vec![Ok::<bytes::Bytes, String>(bytes::Bytes::from(events))]);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

        let result = parse_sse_stream(stream, tx).await.expect("should parse");

        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(
            result.tool_calls[0].reasoning.as_deref(),
            Some("I should search")
        );
    }
}
