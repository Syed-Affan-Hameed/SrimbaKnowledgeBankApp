export interface ConversationMessage {
  sender: "user" | "AI-bot";
  message: string;
}

export type ConversationHistory = ConversationMessage[];