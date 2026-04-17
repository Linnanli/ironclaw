//! Git tools for version control operations.
//!
//! Provides typed tools that shell out to `git` CLI with risk-level classification:
//!
//! | Tool            | Risk   | Approval          |
//! |-----------------|--------|-------------------|
//! | git_status      | Low    | Never             |
//! | git_diff        | Low    | Never             |
//! | git_log         | Low    | Never             |
//! | git_branch      | Medium | UnlessAutoApproved|
//! | git_commit      | Medium | UnlessAutoApproved|
//! | git_push        | High   | Always            |

mod runner;
mod status;
mod diff;
mod log;
mod commit;
mod branch;
mod push;

pub use status::GitStatusTool;
pub use diff::GitDiffTool;
pub use log::GitLogTool;
pub use commit::GitCommitTool;
pub use branch::GitBranchTool;
pub use push::GitPushTool;
