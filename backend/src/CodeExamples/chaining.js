import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";

dotenv.config();

const openAIApiKey = process.env.OPENAI_API_KEY;

const llm = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0,
    openAIApiKey: openAIApiKey,
  });

const tweetTemplate = 'Generate a promotional tweet for a product, from this product description: {productDesc}'

const tweetPrompt = ChatPromptTemplate.fromTemplate(tweetTemplate)

const tweetChain = tweetPrompt.pipe(llm).pipe(new StringOutputParser());

const response = await tweetChain.invoke({productDesc: 'Electric shoes'});

console.log(response)
