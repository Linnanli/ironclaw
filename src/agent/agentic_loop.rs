//! Unified agentic loop engine — re-export shim.
//!
//! Phase 3 Step D-4.5: the real engine moved to
//! [`x_claw_agent::agentic_loop`]. This module now only re-exports the
//! public surface so existing `use crate::agent::agentic_loop::{...}` call
//! sites (in [`crate::agent::dispatcher`], [`crate::worker::job`],
//! [`crate::worker::container`]) keep resolving unchanged.
//!
//! Route-B: `LoopDelegate::call_llm` and `run_agentic_loop` no longer take
//! `reasoning: &Reasoning`. Each delegate now owns a `Reasoning` engine as a
//! field and uses `self.reasoning` directly. Error type at the trait
//! boundary is `x_claw_agent::traits::HostError`
//! (`Box<dyn std::error::Error + Send + Sync>`); ironclaw's internal
//! helpers keep returning `crate::error::Error` and rely on the blanket
//! `From<E> for Box<dyn Error + Send + Sync>` to cross the boundary via
//! `?` or `.map_err(Into::into)`.

pub use x_claw_agent::agentic_loop::{
    AgenticLoopConfig, LoopDelegate, LoopOutcome, LoopSignal, TextAction, run_agentic_loop,
};
pub use x_claw_agent::intent::truncate_for_preview;

use x_claw_agent::traits::HostError;

/// Convert a `HostError` produced by the engine back into ironclaw's
/// concrete [`crate::error::Error`].
///
/// Delegates in this crate always return errors that originated as
/// `crate::error::Error` and were boxed via the blanket
/// `From<E> for Box<dyn Error + Send + Sync>`. We therefore downcast
/// first, and only wrap as `LlmError::InvalidResponse` if the boxed error
/// came from somewhere else (shouldn't happen in practice).
pub(crate) fn host_err_to_error(e: HostError) -> crate::error::Error {
    match e.downcast::<crate::error::Error>() {
        Ok(boxed) => *boxed,
        Err(other) => crate::error::LlmError::InvalidResponse {
            provider: "agent".to_string(),
            reason: other.to_string(),
        }
        .into(),
    }
}

/// Build an `x_claw_agent::HookBundle` whose `safety` slot is wired to
/// the ironclaw [`crate::safety::SafetyLayer`] via
/// [`ironclaw_safety::agent_hook::IronclawSafetyHook`].
///
/// `sandbox` / `secrets` / `approval` keep their `Noop` / `InMemory` /
/// `AutoApprove` defaults — they will be wired in by later Phase 3 steps.
/// This helper exists so every consumer (chat dispatcher, job worker,
/// container worker) constructs an identical bundle and we do not lose
/// the safety boundary the moment any one call site forgets to plug it in.
pub fn hook_bundle_with_safety(
    safety: std::sync::Arc<crate::safety::SafetyLayer>,
) -> x_claw_agent::HookBundle {
    let mut bundle = x_claw_agent::HookBundle::noop();
    bundle.safety = std::sync::Arc::new(
        ironclaw_safety::agent_hook::IronclawSafetyHook::new(safety),
    );
    bundle
}
