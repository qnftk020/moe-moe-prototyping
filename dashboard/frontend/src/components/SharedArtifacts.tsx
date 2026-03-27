"use client";

import { Artifact } from "@/lib/types";

interface Props {
  artifacts: Artifact[];
}

const ICON_BG: Record<string, string> = {
  md: "bg-[#e8f0fe]",
  code: "bg-[#fbecdd]",
  review: "bg-[#dbeddb]",
};

const ICON_EMOJI: Record<string, string> = {
  md: "📋",
  code: "💻",
  review: "📝",
};

const TAG_STYLE: Record<string, string> = {
  sonnet: "bg-[#eae4f2] text-[#6940a5]",
  claude: "bg-[#fbecdd] text-[#d9730d]",
};

export default function SharedArtifacts({ artifacts }: Props) {
  return (
    <div className="border border-[#e9e9e7] rounded-lg overflow-hidden">
      <div className="flex items-center gap-1.5 px-4 py-2.5 text-[13px] font-semibold border-b border-[#e9e9e7] bg-white text-[#787774]">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
          <path d="M3 1a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2V3a2 2 0 00-2-2H3zm1 3h8v1.5H4V4zm0 3h6v1.5H4V7zm0 3h8v1.5H4V10z" />
        </svg>
        Shared artifacts
      </div>
      <div className="px-3 py-2 bg-white max-h-[300px] overflow-y-auto">
        {artifacts.length === 0 ? (
          <div className="text-[#b4b4b0] text-center py-4 text-sm">
            아직 산출물이 없습니다
          </div>
        ) : (
          artifacts.map((artifact, i) => (
            <div
              key={i}
              className="flex items-center gap-2.5 px-3 py-2.5 rounded-md mb-1.5 cursor-pointer transition-all border border-transparent hover:bg-[#f7f7f5] hover:border-[#e9e9e7]"
            >
              <div
                className={`w-9 h-9 rounded-md flex items-center justify-center text-base flex-shrink-0 ${
                  ICON_BG[artifact.icon_type] || ICON_BG.code
                }`}
              >
                {ICON_EMOJI[artifact.icon_type] || "📄"}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-[13px] font-semibold truncate">
                  {artifact.title}
                </div>
                <div className="flex gap-2 text-[11px] text-[#787774]">
                  <span>{artifact.size}</span>
                  <span>{artifact.created_at}</span>
                </div>
              </div>
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded font-semibold ${
                  TAG_STYLE[artifact.created_by] || ""
                }`}
              >
                {artifact.created_by === "sonnet" ? "Sonnet" : "Opus"}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
