"use client";

import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import { LogEntry } from "@/lib/types";

interface StructuredQuestion {
  id: string;
  text: string;
  options: string[];
  multi_select: boolean;
}

interface Props {
  agent: "sonnet" | "claude";
  logs: LogEntry[];
  status: "running" | "idle" | "waiting";
  onSendMessage: (message: string) => void;
  structuredQuestion?: StructuredQuestion | null;
}

const PREFIX_COLORS: Record<string, string> = {
  SNT: "text-[#2383e2]",
  OPS: "text-[#d9730d]",
  SYS: "text-[#0f7b6c]",
  USR: "text-[#6940a5]",
  ERR: "text-[#e03e3e]",
};

const BADGE_CONFIG = {
  sonnet: {
    emoji: "✨",
    label: "Sonnet",
    badgeClass: "bg-[#eae4f2] text-[#6940a5]",
    name: "Planner (Sonnet)",
  },
  claude: {
    emoji: "⚡",
    label: "Claude",
    badgeClass: "bg-[#fbecdd] text-[#d9730d]",
    name: "Builder (Opus)",
  },
};

// Prefixes that should be grouped when consecutive
const GROUPABLE = new Set(["SNT", "OPS"]);
const PREVIEW_LINES = 3;

/** Group consecutive GEM/CLD lines into collapsible blocks */
interface LogGroup {
  type: "single" | "block";
  logs: LogEntry[];
}

function groupLogs(logs: LogEntry[]): LogGroup[] {
  const groups: LogGroup[] = [];
  let i = 0;

  while (i < logs.length) {
    const log = logs[i];

    if (GROUPABLE.has(log.prefix)) {
      // Collect consecutive lines with the same groupable prefix
      const block: LogEntry[] = [log];
      while (
        i + 1 < logs.length &&
        logs[i + 1].prefix === log.prefix
      ) {
        i++;
        block.push(logs[i]);
      }

      if (block.length > PREVIEW_LINES + 1) {
        groups.push({ type: "block", logs: block });
      } else {
        // Small group — show individually
        for (const entry of block) {
          groups.push({ type: "single", logs: [entry] });
        }
      }
    } else {
      groups.push({ type: "single", logs: [log] });
    }
    i++;
  }

  return groups;
}

/* Options are now received via structuredQuestion prop — no text parsing needed */

function InteractiveQA({
  options,
  multiSelect,
  onSelect,
  onSendMessage,
}: {
  options: string[];
  multiSelect: boolean;
  onSelect: (value: string) => void;
  onSendMessage: (message: string) => void;
}) {
  const [showFreeInput, setShowFreeInput] = useState(false);
  const [freeText, setFreeText] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const toggleItem = (opt: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(opt)) next.delete(opt);
      else next.add(opt);
      return next;
    });
  };

  const submitMulti = () => {
    if (selected.size > 0) {
      onSelect(Array.from(selected).join(", "));
      setSelected(new Set());
    }
  };

  // If options detected → show option buttons
  if (options.length > 0 && !showFreeInput) {
    return (
      <div className="border-t border-[#d9730d] bg-gradient-to-b from-[#fbecdd]/40 to-white shrink-0">
        {/* Option buttons */}
        <div className="px-4 pt-3 pb-2">
          <div className="text-[11px] text-[#d9730d] font-semibold mb-2 flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#d9730d] animate-pulse" />
            {multiSelect ? "복수 선택 가능 — 선택 후 전송 버튼" : "선택해주세요"}
          </div>
          <div className="flex flex-wrap gap-2">
            {options.map((opt, i) => {
              const isSelected = selected.has(opt);
              return (
                <button
                  key={i}
                  onClick={() => multiSelect ? toggleItem(opt) : onSelect(opt)}
                  className={`px-4 py-2 text-[12px] font-medium border-2 rounded-lg transition-all active:scale-95 ${
                    isSelected
                      ? "bg-[#d9730d] text-white border-[#d9730d] shadow-md"
                      : "bg-white border-[#e9e9e7] text-[#37352f] hover:border-[#d9730d] hover:bg-[#fbecdd]/30 hover:shadow-md"
                  }`}
                >
                  {multiSelect && isSelected && "✓ "}{opt}
                </button>
              );
            })}
          </div>
          {/* Multi-select submit button */}
          {multiSelect && selected.size > 0 && (
            <button
              onClick={submitMulti}
              className="mt-2 px-5 py-2 text-[12px] font-semibold bg-[#d9730d] text-white rounded-lg hover:opacity-90 transition-opacity"
            >
              선택 완료 ({selected.size}개)
            </button>
          )}
        </div>
        {/* "직접 입력" option */}
        <div className="px-4 pb-3">
          <button
            onClick={() => setShowFreeInput(true)}
            className="text-[11px] text-[#787774] hover:text-[#d9730d] transition-colors"
          >
            직접 입력하기 →
          </button>
        </div>
      </div>
    );
  }

  // Free text input mode (after clicking "직접 입력" or when no options detected)
  return (
    <div className="border-t border-[#d9730d] bg-gradient-to-b from-[#fbecdd]/40 to-white shrink-0">
      <div className="px-4 py-3">
        <div className="text-[11px] text-[#d9730d] font-semibold mb-2 flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-[#d9730d] animate-pulse" />
          답변을 입력해주세요
        </div>
        <div className="flex gap-2">
          <input
            className="flex-1 px-3 py-2 text-[13px] border-2 border-[#d9730d]/30 rounded-lg outline-none bg-white text-[#37352f] placeholder:text-[#b4b4b0] focus:border-[#d9730d] focus:shadow-[0_0_0_3px_rgba(217,115,13,0.12)]"
            placeholder="답변을 입력하세요..."
            value={freeText}
            onChange={(e) => setFreeText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && freeText.trim()) {
                onSendMessage(freeText.trim());
                setFreeText("");
                setShowFreeInput(false);
              }
            }}
            autoFocus
          />
          <button
            className="px-4 py-2 text-xs font-semibold bg-[#d9730d] text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50"
            onClick={() => {
              if (freeText.trim()) {
                onSendMessage(freeText.trim());
                setFreeText("");
                setShowFreeInput(false);
              }
            }}
            disabled={!freeText.trim()}
          >
            전송
          </button>
        </div>
        {options.length > 0 && (
          <button
            onClick={() => setShowFreeInput(false)}
            className="text-[11px] text-[#787774] hover:text-[#d9730d] mt-2 transition-colors"
          >
            ← 선택지로 돌아가기
          </button>
        )}
      </div>
    </div>
  );
}

function LogLine({ log }: { log: LogEntry }) {
  const isUserMsg = log.prefix === "USR";
  return (
    <div className="flex gap-2 py-0.5">
      <span className="text-[#b4b4b0] flex-shrink-0 text-[11px] min-w-[52px] select-none">
        {log.timestamp?.slice(0, 5) || ""}
      </span>
      <span
        className={`flex-shrink-0 font-semibold ${
          PREFIX_COLORS[log.prefix] || "text-[#787774]"
        }`}
      >
        {log.prefix}
      </span>
      <span
        className={`break-words ${
          isUserMsg
            ? "bg-[rgba(35,131,226,0.08)] px-1 rounded"
            : log.prefix === "SYS"
            ? "text-[#787774]"
            : ""
        }`}
      >
        {log.content}
      </span>
    </div>
  );
}

function CollapsibleBlock({ group }: { group: LogGroup }) {
  const [expanded, setExpanded] = useState(false);
  const logs = group.logs;
  const prefix = logs[0].prefix;
  const hiddenCount = logs.length - PREVIEW_LINES;

  // Build a one-line summary from the first meaningful line
  const summary = logs[0].content.length > 60
    ? logs[0].content.slice(0, 57) + "..."
    : logs[0].content;

  return (
    <div className="my-1 rounded-md border border-[#e9e9e7] bg-white/60 overflow-hidden">
      {/* Preview: first N lines */}
      <div className="px-2 py-1">
        {logs.slice(0, expanded ? logs.length : PREVIEW_LINES).map((log, i) => (
          <LogLine key={i} log={log} />
        ))}
      </div>

      {/* Toggle button */}
      {!expanded ? (
        <button
          className="w-full px-3 py-1.5 text-[11px] text-[#787774] hover:text-[#37352f] hover:bg-[#f1f1ef] border-t border-[#e9e9e7] transition-colors text-left"
          onClick={() => setExpanded(true)}
        >
          ▶ {hiddenCount}줄 더 보기
        </button>
      ) : (
        <button
          className="w-full px-3 py-1.5 text-[11px] text-[#787774] hover:text-[#37352f] hover:bg-[#f1f1ef] border-t border-[#e9e9e7] transition-colors text-left"
          onClick={() => setExpanded(false)}
        >
          ▲ 접기
        </button>
      )}
    </div>
  );
}

export default function AgentPanel({ agent, logs, status, onSendMessage, structuredQuestion }: Props) {
  const logEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [input, setInput] = useState("");
  const config = BADGE_CONFIG[agent];
  const prevStatusRef = useRef(status);

  const groups = useMemo(() => groupLogs(logs), [logs]);

  // No auto-scroll — user has full control

  const handleSend = () => {
    if (!input.trim()) return;
    onSendMessage(input.trim());
    setInput("");
  };

  const isRunning = status === "running";
  const isWaiting = status === "waiting";

  // Options come from structuredQuestion prop — no parsing needed

  const handleQuickSelect = useCallback((value: string) => {
    onSendMessage(value);
    setInput("");
  }, [onSendMessage]);

  const statusDotClass = isWaiting
    ? "bg-[#d9730d] animate-pulse"
    : isRunning
    ? "bg-[#0f7b6c] animate-pulse"
    : "bg-[#b4b4b0]";
  const statusLabel = isWaiting ? "Waiting for input" : isRunning ? "Running" : "Idle";

  return (
    <div className={`border rounded-lg overflow-hidden flex flex-col min-h-0 h-full ${
      isWaiting ? "border-[#d9730d] shadow-[0_0_0_2px_rgba(217,115,13,0.15)]" : "border-[#e9e9e7]"
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#e9e9e7] bg-white shrink-0">
        <div className="flex items-center gap-2.5">
          <span
            className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold ${config.badgeClass}`}
          >
            {config.emoji} {config.label}
          </span>
          <span className="text-sm font-semibold">{config.name}</span>
        </div>
        <div className={`flex items-center gap-1.5 text-xs ${isWaiting ? "text-[#d9730d] font-semibold" : "text-[#787774]"}`}>
          <div className={`w-1.5 h-1.5 rounded-full ${statusDotClass}`} />
          {statusLabel}
        </div>
      </div>

      {/* Log Body — always scrollable */}
      <div
        ref={scrollContainerRef}
        className="flex-1 min-h-0 bg-[#f7f7f5] overflow-y-auto"
      >
        <div className="px-4 py-3 font-mono text-[12.5px] leading-[1.8]">
          {groups.length === 0 && (
            <div className="text-[#b4b4b0] text-center py-8">대기 중...</div>
          )}
          {groups.map((group, i) => {
            if (group.type === "block") {
              return <CollapsibleBlock key={i} group={group} />;
            }

            const log = group.logs[0];
            const prevGroup = groups[i - 1];
            const prevLog = prevGroup?.logs[prevGroup.logs.length - 1];
            const showDivider =
              log.prefix === "SYS" && prevLog && prevLog.prefix !== "SYS";

            return (
              <div key={i}>
                {showDivider && (
                  <hr className="border-t border-dashed border-[#e9e9e7] my-2" />
                )}
                <LogLine log={log} />
              </div>
            );
          })}
          <div ref={logEndRef} />
        </div>
      </div>

      {/* Running indicator bar */}
      {isRunning && (
        <div className="h-1 bg-gradient-to-r from-transparent via-[#2383e2] to-transparent animate-pulse" />
      )}

      {/* Interactive Footer */}
      {isWaiting ? (
        <InteractiveQA
          options={structuredQuestion?.options || []}
            multiSelect={structuredQuestion?.multi_select || false}
          onSelect={handleQuickSelect}
          onSendMessage={onSendMessage}
        />
      ) : (
        /* Minimal footer when not in Q&A mode */
        <div className="flex items-center gap-2 px-4 py-2 border-t border-[#e9e9e7] bg-white shrink-0">
          <input
            className="flex-1 px-2.5 py-1.5 text-[13px] border border-[#e9e9e7] rounded bg-[#f7f7f5] outline-none focus:border-[#2383e2] focus:bg-white text-[#37352f] placeholder:text-[#b4b4b0]"
            placeholder={`${config.label}에게 추가 지시...`}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />
          <button
            className="px-3 py-1.5 text-xs font-semibold border border-[#e9e9e7] rounded bg-white text-[#37352f] hover:bg-[#f1f1ef] transition-colors"
            onClick={handleSend}
          >
            Send
          </button>
        </div>
      )}
    </div>
  );
}
