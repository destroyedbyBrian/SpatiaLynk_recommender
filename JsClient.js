import { Client } from "@huggingface/client";


// .connect() 
    //  -> arguments : source (url) & options (token + status_callback)
    //  -> returns : an obj that allows you to make calls to the API
const app = await Client.connect("destroyedbyBrian/spatiaLynk_recommender", {
  token: process.env.HF_TOKEN,
  status_callback: (status) => {
    console.log("Status:", status);
  } 
});

// .predict()
    // -> arguments : endpoint & payload
const result = await app.predict("recommend", {
    userId: "user_123",
    itemId: "item_456",
    topK: 5
})

// .submit()    
    // - similar to .predict() but does not return a promise and should not be awaited,
    // instead returns an async iterator with a cancel method.

    