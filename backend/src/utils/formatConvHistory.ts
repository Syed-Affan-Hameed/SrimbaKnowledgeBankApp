import { ConversationHistory } from "../models/conversationHistoryModel.js";

export const formatConvHistory = (convHistory: ConversationHistory): string => {
  return convHistory
    .map((entry) => `${entry.sender}: ${entry.message}`)
    .join("\n");
};
