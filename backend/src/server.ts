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

app.get("/generateStandaloneQuestion", async (req, res) => {
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
  const embeddingClassInstance = new OpenAIEmbeddings({openAIApiKey: openAIApiKey});
  const client = createClient(supabaseUrl!, supabaseApiKey!);

  // setting up the supabase vector store as retriever

  const vectorStore = new SupabaseVectorStore(embeddingClassInstance,{client,tableName:"documents",queryName:"match_documents"});

  const retriever = vectorStore.asRetriever();
  
  const standaloneQuestionTemplate =
    "Generate a standalone question from this user-Prompt: {userPrompt}";

  const standaloneQuestionPrompt = ChatPromptTemplate.fromTemplate(
    standaloneQuestionTemplate
  );

  const standaloneQuestionChain = standaloneQuestionPrompt
    .pipe(llmModel)
    .pipe(new StringOutputParser());

  const response = await standaloneQuestionChain.invoke({
    userPrompt: userPrompt,
  });

  console.log(response);

  res.send(response);
});
