import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createClient } from '@supabase/supabase-js'

dotenv.config();
const app = express();

const PORT = process.env.PORT || 5005;
//middleware to use json throughout our application
app.use(express.json());
// Define allowed origins
const allowedOrigins = [
    "http://localhost:5173",
    "http://localhost:5005", 
  ];
  
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
  app.listen(PORT,()=>{
    console.log("Server is running in the port "+ PORT);
  });

  app.get("/splitScrimbaData", async (req,res)=>{

    const filePath = "./src/data/scrimba.txt";
    const textFromTheScrimbaDocument = fs.readFileSync(filePath,"utf-8");

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        separators: ['\n\n', '\n', ' ', ''], // default setting
        chunkOverlap: 50
      })
    
      const chunkedText = await splitter.createDocuments([textFromTheScrimbaDocument]);

      const supabaseApiKey = process.env.SUPABASE_API_KEY
      const supabaseUrl = process.env.SUPABASE_URL
      const openAIApiKey = process.env.OPENAI_API_KEY
      
      if (!supabaseApiKey || !supabaseUrl) {
        throw new Error("Supabase API key or URL is not defined");
      }
      const client = createClient(supabaseUrl, supabaseApiKey);
      
      const outputMessage = await SupabaseVectorStore.fromDocuments(
        chunkedText,
          new OpenAIEmbeddings({ openAIApiKey:openAIApiKey }),
          {
             client,
             tableName: 'documents',
         
            }
          )

      console.log(outputMessage);

    res.send(outputMessage);
  });