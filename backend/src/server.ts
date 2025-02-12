import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createClient } from "@supabase/supabase-js";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import {
  conversationHistory,
  getAnswerForOrignialQuestion,
  getAnswerForOrignialQuestionWithConversationHistory,
} from "./utils/conversationHistoryHandler.js";
dotenv.config();
const app = express();

const PORT = process.env.PORT || 5005;
//middleware to use json throughout our application
app.use(express.json());
// Define allowed origins
const allowedOrigins = ["http://localhost:5173", "http://localhost:5005"];

// CORS Middleware
app.use(
  cors({
    origin: function (origin, callback) {
      if (!origin) return callback(null, true);

      if (allowedOrigins.includes(origin)) {
        return callback(null, true);
      } else {
        return callback(new Error("Not allowed by CORS"));
      }
    },
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    credentials: true,
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

// app.use("/api/v1/litSearch",litSearchRouter);
app.listen(PORT, () => {
  console.log("Server is running in the port " + PORT);
});

app.get("/splitScrimbaData", async (req, res) => {
  const filePath = "./src/data/scrimba.txt";
  const textFromTheScrimbaDocument = fs.readFileSync(filePath, "utf-8");

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    separators: ["\n\n", "\n", " ", ""], // default setting
    chunkOverlap: 50,
  });

  const chunkedText = await splitter.createDocuments([
    textFromTheScrimbaDocument,
  ]);

  const supabaseApiKey = process.env.SUPABASE_API_KEY;
  const supabaseUrl = process.env.SUPABASE_URL;
  const openAIApiKey = process.env.OPENAI_API_KEY;

  if (!supabaseApiKey || !supabaseUrl) {
    throw new Error("Supabase API key or URL is not defined");
  }
  const client = createClient(supabaseUrl, supabaseApiKey);

  const outputMessage = await SupabaseVectorStore.fromDocuments(
    chunkedText,
    new OpenAIEmbeddings({ openAIApiKey: openAIApiKey }),
    {
      client,
      tableName: "documents",
    }
  );

  console.log(outputMessage);

  res.send(outputMessage);
});
app.get("/getProductDesc", async (req, res) => {
  const llm = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const tweetTemplate =
    "Generate a promotional tweet for a product, from this product description: {productDesc}";

  const tweetPrompt = ChatPromptTemplate.fromTemplate(tweetTemplate);

  const tweetChain = tweetPrompt.pipe(llm).pipe(new StringOutputParser());

  const response = await tweetChain.invoke({ productDesc: "Electric shoes" });

  console.log(response);
  res.send(response);
});

app.get(
  "/generateStandaloneQuestionAndRetrieveEmbeddings",
  async (req, res) => {
    const userPrompt = req.body.userPrompt;
    const llmModel = new ChatOpenAI({
      model: "gpt-3.5-turbo",
      temperature: 0,
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    //creating client for supabase
    const supabaseApiKey = process.env.SUPABASE_API_KEY;
    const supabaseUrl = process.env.SUPABASE_URL;
    const openAIApiKey = process.env.OPENAI_API_KEY;
    const embeddingClassInstance = new OpenAIEmbeddings({
      openAIApiKey: openAIApiKey,
    });
    const client = createClient(supabaseUrl!, supabaseApiKey!);

    // setting up the supabase vector store as retriever

    const vectorStore = new SupabaseVectorStore(embeddingClassInstance, {
      client,
      tableName: "documents",
      queryName: "match_documents",
    });

    const retriever = vectorStore.asRetriever();

    const standaloneQuestionTemplate =
      "Generate a standalone question from this user-Prompt: {userPrompt}";

    const standaloneQuestionPrompt = ChatPromptTemplate.fromTemplate(
      standaloneQuestionTemplate
    );

    // this will retrieve the nearest matching vectors from the supabase
    const Chain = standaloneQuestionPrompt
      .pipe(llmModel)
      .pipe(new StringOutputParser())
      .pipe(retriever);

    const response = await Chain.invoke({
      userPrompt: userPrompt,
    });

    const context = response[0].pageContent;

    console.log(response);

    res.send(response);
  }
);

app.post("/translateSentence", async (req, res) => {
  const userSentence = req.body.userSentence;
  const translationLanguage = req.body.translationLanguage;
  try {
    const translatedSentence = await translateSentence(
      userSentence,
      translationLanguage
    );

    res.send({
      message: "Translation successfull",
      Translated_Sentence: translatedSentence,
    });
  } catch (error: any) {
    res
      .status(500)
      .send({ message: "Translation failed", error: error.message });
    return;
  }
});

const translateSentence = async (
  userSentence: string,
  translationLanguage: string
) => {
  const llmModel = new ChatOpenAI({
    model: "gpt-4",
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const punctuationTemplate = `Given a sentence, add punctuation where needed. 
    sentence: {sentence}
    sentence with punctuation:  
    `;
  const punctuationPrompt =
    ChatPromptTemplate.fromTemplate(punctuationTemplate);

  const grammarTemplate = `Given a sentence correct the grammar.
    sentence: {punctuated_sentence}
    sentence with correct grammar: 
    `;
  const grammarPrompt = ChatPromptTemplate.fromTemplate(grammarTemplate);

  const translationTemplate = `Given a sentence, translate that sentence into {language}
    sentence: {grammatically_correct_sentence}
    translated sentence:
    `;
  const translationPrompt =
    ChatPromptTemplate.fromTemplate(translationTemplate);

  // const punctuationChain = RunnableSequence.from([
  //     punctuationPrompt,
  //     llmModel,
  //     new StringOutputParser()
  // ])
  // const grammarChain = RunnableSequence.from([
  //     grammarPrompt,
  //     llmModel,
  //     new StringOutputParser()
  // ])
  // const translationChain = RunnableSequence.from([
  //     translationPrompt,
  //     llmModel,
  //     new StringOutputParser()
  // ])

  const punctuationChain = punctuationPrompt
    .pipe(llmModel)
    .pipe(new StringOutputParser());
  const grammarChain = grammarPrompt
    .pipe(llmModel)
    .pipe(new StringOutputParser());
  const translationChain = translationPrompt
    .pipe(llmModel)
    .pipe(new StringOutputParser());
  const chain = RunnableSequence.from([
    {
      punctuated_sentence: punctuationChain,
      original_input: new RunnablePassthrough(),
    },
    {
      grammatically_correct_sentence: grammarChain,
      language: (input) => input.original_input.language,
    },
    translationChain,
  ]);

  const response = await chain.invoke({
    sentence: userSentence,
    language: translationLanguage,
  });

  console.log(response);

  return response;
};

app.post("/srimbaBot", async (req, res) => {
  const originalQuestion = req.body.userQuestion;

  const llmModel = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const supabaseApiKey = process.env.SUPABASE_API_KEY;
  const supabaseUrl = process.env.SUPABASE_URL;
  const openAIApiKey = process.env.OPENAI_API_KEY;
  const embeddingClassInstance = new OpenAIEmbeddings({
    openAIApiKey: openAIApiKey,
  });

  const supabaseClient = createClient(supabaseUrl!, supabaseApiKey!);
  const vectorStore = new SupabaseVectorStore(embeddingClassInstance, {
    client: supabaseClient,
    tableName: "documents",
    queryName: "match_documents",
  });

  const retriever = vectorStore.asRetriever();

  const standaloneQuestionTemplate = `Generate a standalone question from this user-Prompt: {originalQuestion}`;

  const standaloneQuestionPrompt = ChatPromptTemplate.fromTemplate(
    standaloneQuestionTemplate
  );

  const contextArrayChain = standaloneQuestionPrompt
    .pipe(llmModel)
    .pipe(new StringOutputParser())
    .pipe(retriever);

  const response = await contextArrayChain.invoke({ originalQuestion });

  const consolidatedContext = response
    .map((context) => context.pageContent)
    .join(" ");

  const answer = await getAnswerForOrignialQuestion(
    originalQuestion,
    llmModel,
    consolidatedContext
  );

  console.log(answer);

  res.send(answer);
});

app.post("/srimbaBotWithConversationHistory", async (req, res) => {

  const originalQuestion = req.body.userQuestion;

  conversationHistory.push({
    sender: "user",
    message: originalQuestion,
  });
  let conversationHistoryUserString = JSON.stringify(conversationHistory);

  const llmModel = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const supabaseApiKey = process.env.SUPABASE_API_KEY;
  const supabaseUrl = process.env.SUPABASE_URL;
  const openAIApiKey = process.env.OPENAI_API_KEY;
  const embeddingClassInstance = new OpenAIEmbeddings({
    openAIApiKey: openAIApiKey,
  });

  const supabaseClient = createClient(supabaseUrl!, supabaseApiKey!);
  const vectorStore = new SupabaseVectorStore(embeddingClassInstance, {
    client: supabaseClient,
    tableName: "documents",
    queryName: "match_documents",
  });

  const retriever = vectorStore.asRetriever();

  const standaloneQuestionTemplate = `Given some conversation history (if any) and a question, convert the question to a standalone question. 
conversation history: {conversationHistory}
question: {originalQuestion} 
standalone question:`;

  const standaloneQuestionPrompt = ChatPromptTemplate.fromTemplate(
    standaloneQuestionTemplate
  );

  const contextArrayChain = standaloneQuestionPrompt
    .pipe(llmModel)
    .pipe(new StringOutputParser())
    .pipe(retriever);

  const response = await contextArrayChain.invoke({
    originalQuestion,
    conversationHistory: conversationHistoryUserString,
  });

  const consolidatedContext = response
    .map((context) => context.pageContent)
    .join(" ");

  const answer = await getAnswerForOrignialQuestionWithConversationHistory(
    originalQuestion,
    llmModel,
    consolidatedContext,
    conversationHistory
  );

  conversationHistory.push({
    sender: "AI-bot",
    message: answer,
  });

  
  console.log(answer);

  res.send(answer);
});
