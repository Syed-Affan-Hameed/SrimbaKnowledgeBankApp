import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";

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

