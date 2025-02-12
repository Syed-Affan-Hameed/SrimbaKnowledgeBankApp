import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { ConversationHistory } from "../models/conversationHistoryModel.js";

  export const getAnswerForOrignialQuestion = async (
    userQuestion: string,
    llm: ChatOpenAI,
    context: string
  ): Promise<any> => {
    const anserForOriginalQuestionTemplate = `Generate an answer for this user question: {userQuestion} for the given context: {context} remeber the rules   - be friendly only answer from the context provided and never make up answers
  apologise if it doesn't know the answer and advise the 
    user to email help@scrimba.com`;
  
    const answerPrompt = ChatPromptTemplate.fromTemplate(
      anserForOriginalQuestionTemplate
    );
  
    const chain = answerPrompt.pipe(llm).pipe(new StringOutputParser());
  
    const response = chain.invoke({ userQuestion, context });
  
    return response;
  };

  export const getAnswerForOrignialQuestionWithConversationHistory = async (
    userQuestion: string,
    llm: ChatOpenAI,
    context: string,
    conversationHistory: ConversationHistory
  ): Promise<any> => {

    let conversationHistoryString = JSON.stringify(conversationHistory);
  
    const answerForOriginalQuestionTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Scrimba based on the context provided and the conversation history. Try to find the answer in the context. If the answer is not given in the context, find the answer in the conversation history if possible. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@scrimba.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
context: {context}
conversation history: {conversation-history}
question: {userQuestion}
answer: `
    const answerPrompt = ChatPromptTemplate.fromTemplate(
      answerForOriginalQuestionTemplate
    );
  
    const chain = answerPrompt.pipe(llm).pipe(new StringOutputParser());
  
    console.log("Conversation History: ",conversationHistoryString);

    const response = chain.invoke({ userQuestion, context, 'conversation-history': conversationHistoryString });

    return response;
  };

  export let conversationHistory :ConversationHistory = [];