"use client";

import type { UIMessage, UseChatHelpers } from "@ai-sdk/react";
import { motion } from "framer-motion";
import { Streamdown } from "streamdown";

import { TepsoutIcon } from "./icons";
import { PreviewAttachment } from "./preview-attachment";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";

export const PreviewMessage = ({
  message,
  isLoading,
  isLast,
  sendMessage,
}: {
  chatId: string;
  message: UIMessage;
  isLoading: boolean;
  isLast?: boolean;
  sendMessage?: UseChatHelpers<UIMessage>["sendMessage"];
}) => {

  return (
    <motion.div
      className="w-full mx-auto max-w-3xl px-4 group/message"
      initial={{ y: 5, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      data-role={message.role}
    >
      <div
        className={cn(
          "group-data-[role=user]/message:bg-primary group-data-[role=user]/message:text-primary-foreground flex gap-4 group-data-[role=user]/message:px-3 w-full group-data-[role=user]/message:w-fit group-data-[role=user]/message:ml-auto group-data-[role=user]/message:max-w-2xl group-data-[role=user]/message:py-2 rounded-xl"
        )}
      >
        {message.role === "assistant" && (
          <div className="size-8 flex items-center rounded-full justify-center ring-1 shrink-0 ring-border">
            <TepsoutIcon size={28} />
          </div>
        )}

        <div className="flex flex-col gap-2 w-full">
          {message.parts &&
            message.parts.map((part: any, index: number) => {
              if (part.type === "text") {
                return (
                  <div key={index} className="flex flex-col gap-4">
                    <Streamdown>{part.text}</Streamdown>
                  </div>
                );
              }
              // Handle tool calls - type is "tool-{toolName}" in AI SDK v5
              if (part.type?.startsWith("tool-")) {
                const { toolCallId, state, output } = part;
                const toolName = part.type.replace("tool-", "");

                if (state === "output-available" && output) {
                  return (
                    <div key={toolCallId}>
                      <pre>{JSON.stringify(output, null, 2)}</pre>
                    </div>
                  );
                }
              }
              if (part.type === "file") {
                return (
                  <PreviewAttachment
                    key={index}
                    attachment={part}
                  />
                );
              }
              if (part.type === "data-sources" && Array.isArray(part.data)) {
                return (
                  <div key={index} className="flex flex-col items-start gap-1 mt-2">
                    <span className="text-sm text-muted-foreground">
                      Sources :
                    </span>
                    {part.data.map((source: any, sourceIndex: number) => {
                      const pages =
                        source.page_start &&
                        source.page_end &&
                        source.page_start !== source.page_end
                          ? `, p. ${source.page_start}–${source.page_end}`
                          : source.page_start
                            ? `, p. ${source.page_start}`
                            : "";
                      const label = `[${source.index}] ${source.file_name}${pages}`;
                      return source.link_preview ? (
                        <a
                          key={`source-${sourceIndex}`}
                          href={source.link_preview}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm underline underline-offset-4 hover:text-muted-foreground"
                        >
                          {label}
                        </a>
                      ) : (
                        <span key={`source-${sourceIndex}`} className="text-sm">
                          {label}
                        </span>
                      );
                    })}
                  </div>
                );
              }
              if (
                part.type === "data-questions" &&
                isLast &&
                sendMessage &&
                Array.isArray(part.data)
              ) {
                return (
                  <div key={index} className="flex flex-col items-start gap-2 mt-2">
                    <span className="text-sm text-muted-foreground">
                      Pour approfondir :
                    </span>
                    {part.data.map((question: string, questionIndex: number) => (
                      <Button
                        key={`followup-${questionIndex}`}
                        variant="outline"
                        size="sm"
                        disabled={isLoading}
                        onClick={() => sendMessage({ text: question })}
                        className="h-auto whitespace-normal text-left justify-start py-2"
                      >
                        {question}
                      </Button>
                    ))}
                  </div>
                );
              }
              return null;
            })}
        </div>
      </div>
    </motion.div>
  );
};

export const ThinkingMessage = () => {
  const role = "assistant";

  return (
    <motion.div
      className="w-full mx-auto max-w-3xl px-4 group/message "
      initial={{ y: 5, opacity: 0 }}
      animate={{ y: 0, opacity: 1, transition: { delay: 1 } }}
      data-role={role}
    >
      <div
        className={cn(
          "flex gap-4 group-data-[role=user]/message:px-3 w-full group-data-[role=user]/message:w-fit group-data-[role=user]/message:ml-auto group-data-[role=user]/message:max-w-2xl group-data-[role=user]/message:py-2 rounded-xl",
          {
            "group-data-[role=user]/message:bg-muted": true,
          }
        )}
      >
        <div className="size-8 flex items-center rounded-full justify-center ring-1 shrink-0 ring-border">
          <TepsoutIcon size={28} />
        </div>

        <div className="flex flex-col gap-2 w-full">
          <div className="flex flex-col gap-4 text-muted-foreground">
            Je réfléchis...
          </div>
        </div>
      </div>
    </motion.div>
  );
};
