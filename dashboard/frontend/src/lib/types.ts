export type StepStatus = "waiting" | "active" | "done";

export type PipelineStepName =
  | "envisioning"
  | "blueprinting"
  | "implementation"
  | "review"
  | "feedback";

export interface PipelineState {
  current_step: PipelineStepName;
  steps: Record<PipelineStepName, StepStatus>;
}

export type LogPrefix = "SNT" | "OPS" | "SYS" | "USR" | "ERR";

export interface LogEntry {
  timestamp: string;
  prefix: LogPrefix;
  content: string;
  agent: "sonnet" | "claude";
}

export interface FileNode {
  name: string;
  type: "file" | "directory";
  children: FileNode[];
  is_new: boolean;
}

export interface Artifact {
  title: string;
  description: string;
  file_path: string;
  size: string;
  created_at: string;
  created_by: "sonnet" | "claude";
  icon_type: "md" | "code" | "review";
}

export interface AgentStatus {
  agent: "sonnet" | "claude";
  status: "running" | "idle" | "waiting";
}

export const STEP_LABELS: Record<PipelineStepName, string> = {
  envisioning: "Envisioning",
  blueprinting: "Blueprinting",
  implementation: "Implementation",
  review: "Review",
  feedback: "Feedback Loop",
};

export const STEP_ORDER: PipelineStepName[] = [
  "envisioning",
  "blueprinting",
  "implementation",
  "review",
  "feedback",
];
