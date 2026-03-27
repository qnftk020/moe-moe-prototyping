"use client";

import { useEffect, useState, useCallback } from "react";
import { getSocket } from "@/lib/socket";
import {
  PipelineState,
  LogEntry,
  FileNode,
  Artifact,
  AgentStatus,
} from "@/lib/types";
import Topbar from "@/components/Topbar";
import PipelineStatus from "@/components/PipelineStatus";
import InputArea from "@/components/InputArea";
import AgentPanel from "@/components/AgentPanel";
import ProjectHistory from "@/components/ProjectHistory";
import FileTree from "@/components/FileTree";
import SharedArtifacts from "@/components/SharedArtifacts";
import DevicePreview from "@/components/DevicePreview";
import { Lang, translations } from "@/lib/i18n";

const DEFAULT_PIPELINE: PipelineState = {
  current_step: "envisioning",
  steps: {
    envisioning: "waiting",
    blueprinting: "waiting",
    implementation: "waiting",
    review: "waiting",
    feedback: "waiting",
  },
};

export default function Home() {
  const [lang, setLang] = useState<Lang>("ko");
  const t = translations[lang];
  const [connected, setConnected] = useState(false);
  const [pipelineState, setPipelineState] =
    useState<PipelineState>(DEFAULT_PIPELINE);
  const [sonnetLogs, setSonnetLogs] = useState<LogEntry[]>([]);
  const [claudeLogs, setClaudeLogs] = useState<LogEntry[]>([]);
  const [sonnetStatus, setSonnetStatus] = useState<"running" | "idle" | "waiting">("idle");
  const [claudeStatus, setClaudeStatus] = useState<"running" | "idle" | "waiting">("idle");
  const [fileTree, setFileTree] = useState<FileNode[]>([]);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [appUrl, setAppUrl] = useState<string | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<{
    id: string;
    text: string;
    options: string[];
    multi_select: boolean;
  } | null>(null);

  useEffect(() => {
    const socket = getSocket();

    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));

    socket.on("pipeline_state", (state: PipelineState) => {
      setPipelineState(state);
      const hasActive = Object.values(state.steps).some((s) => s === "active");
      setIsRunning(hasActive);
    });

    socket.on("log", (entry: LogEntry) => {
      if (entry.agent === "sonnet") {
        setSonnetLogs((prev) => [...prev, entry]);
      } else {
        setClaudeLogs((prev) => [...prev, entry]);
      }
    });

    socket.on("agent_status", (data: AgentStatus) => {
      if (data.agent === "sonnet") setSonnetStatus(data.status as "running" | "idle" | "waiting");
      else setClaudeStatus(data.status as "running" | "idle" | "waiting");
    });

    socket.on("waiting_for_input", (data: { agent: string }) => {
      if (data.agent === "sonnet") setSonnetStatus("waiting");
      else setClaudeStatus("waiting");
    });

    socket.on("structured_question", (data: {
      agent: string;
      id: string;
      text: string;
      options: string[];
      multi_select: boolean;
    }) => {
      if (data.agent === "sonnet") {
        setSonnetStatus("waiting");
        setCurrentQuestion(data);
      }
    });

    socket.on("file_tree", (tree: FileNode[]) => setFileTree(tree));
    socket.on("artifact", (artifact: Artifact) => {
      setArtifacts((prev) => [...prev, artifact]);
    });
    socket.on("artifacts_list", (list: Artifact[]) => setArtifacts(list));

    socket.on("app_launched", (data: { url: string }) => {
      setIsRunning(false);
      setAppUrl(data.url);
      window.open(data.url, "_blank");
    });

    return () => {
      socket.off("connect");
      socket.off("disconnect");
      socket.off("pipeline_state");
      socket.off("log");
      socket.off("agent_status");
      socket.off("file_tree");
      socket.off("artifact");
      socket.off("artifacts_list");
      socket.off("waiting_for_input");
      socket.off("structured_question");
      socket.off("app_launched");
    };
  }, []);

  const handleStart = useCallback((prompt: string, googleApiKey?: string) => {
    const socket = getSocket();
    setSonnetLogs([]);
    setClaudeLogs([]);
    setArtifacts([]);
    setFileTree([]);
    setPipelineState(DEFAULT_PIPELINE);
    setIsRunning(true);
    setAppUrl(null);
    socket.emit("start_pipeline", { prompt, google_api_key: googleApiKey || "" });
  }, []);

  const handleStop = useCallback(() => {
    const socket = getSocket();
    socket.emit("stop");
    setIsRunning(false);
    setSonnetStatus("idle");
    setClaudeStatus("idle");
  }, []);

  const handleSendToSonnet = useCallback((message: string) => {
    const socket = getSocket();
    socket.emit("send_to_agent", { agent: "sonnet", message });
    setCurrentQuestion(null);  // Clear question after answering
  }, []);

  const handleSendToClaude = useCallback((message: string) => {
    const socket = getSocket();
    socket.emit("send_to_agent", { agent: "claude", message });
  }, []);

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Topbar connected={connected} onStop={handleStop} lang={lang} onLangChange={setLang} />

      <div className="flex-1 overflow-hidden flex flex-col w-full mx-auto px-4 py-3 sm:px-6 lg:px-10 xl:max-w-[1400px]">
        {/* Title — compact */}
        <div className="flex items-center gap-3 mb-3">
          <span className="text-[28px]">🤖</span>
          <div>
            <h1 className="text-xl font-bold tracking-tight leading-tight">
              {t.title}
            </h1>
            <p className="text-xs text-[#787774]">
              {t.subtitle}
            </p>
          </div>
        </div>

        {/* Pipeline */}
        <PipelineStatus state={pipelineState} />

        {/* Input */}
        <InputArea onStart={handleStart} disabled={isRunning} />

        {/* Agent Panels — fill remaining height, each panel scrolls independently */}
        <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-3 mb-3 min-h-0 overflow-hidden">
          <AgentPanel
            agent="sonnet"
            logs={sonnetLogs}
            status={sonnetStatus}
            onSendMessage={handleSendToSonnet}
            structuredQuestion={currentQuestion}
          />
          <AgentPanel
            agent="claude"
            logs={claudeLogs}
            status={claudeStatus}
            onSendMessage={handleSendToClaude}
          />
        </div>

        {/* Device Preview (Desktop / Tablet / Mobile) */}
        <DevicePreview url={appUrl} />

        {/* Bottom: File Tree + Shared Artifacts + History */}
        <div className="grid grid-cols-1 md:grid-cols-[280px_1fr] gap-3 shrink-0">
          <div>
            <FileTree tree={fileTree} />
            <div className="mt-3">
              <ProjectHistory />
            </div>
          </div>
          <SharedArtifacts artifacts={artifacts} />
        </div>
      </div>
    </div>
  );
}
