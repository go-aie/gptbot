package gptbot_test

import (
	"context"
	"fmt"
	"os"

	"github.com/go-aie/gptbot"
)

func Example() {
	ctx := context.Background()
	apiKey := os.Getenv("OPENAI_API_KEY")
	encoder := gptbot.NewOpenAIEncoder(apiKey, "")
	store := gptbot.NewLocalVectorStore()

	// Feed documents into the vector store.
	feeder := gptbot.NewFeeder(&gptbot.FeederConfig{
		Encoder: encoder,
		Updater: store,
	})
	err := feeder.Feed(ctx, &gptbot.Document{
		ID:   "1",
		Text: "Generative Pre-trained Transformer 3 (GPT-3) is an autoregressive language model released in 2020 that uses deep learning to produce human-like text. Given an initial text as prompt, it will produce text that continues the prompt.\n\nThe architecture is a decoder-only transformer network with a 2048-token-long context and then-unprecedented size of 175 billion parameters, requiring 800GB to store. The model was trained using generative pre-training; it is trained to predict what the next token is based on previous tokens. The model demonstrated strong zero-shot and few-shot learning on many tasks.[2]",
	})
	if err != nil {
		fmt.Printf("err: %v", err)
		return
	}

	// Chat with the bot to get answers.
	bot := gptbot.NewBot(&gptbot.BotConfig{
		APIKey:  apiKey,
		Encoder: encoder,
		Querier: store,
	})

	question := "When was GPT-3 released?"
	answer, err := bot.Chat(ctx, question)
	if err != nil {
		fmt.Printf("err: %v", err)
		return
	}
	fmt.Printf("Q: %s\n", question)
	fmt.Printf("A: %s\n", answer)

	question = "How many parameters does GPT-3 use?"
	answer, err = bot.Chat(ctx, question)
	if err != nil {
		fmt.Printf("err: %v", err)
		return
	}
	fmt.Printf("Q: %s\n", question)
	fmt.Printf("A: %s\n", answer)

	// Output:
	//
	// Q: When was GPT-3 released?
	// A: GPT-3 was released in 2020.
	// Q: How many parameters does GPT-3 use?
	// A: GPT-3 uses 175 billion parameters.
}

func Example_multiTurn() {
	ctx := context.Background()
	apiKey := os.Getenv("OPENAI_API_KEY")
	encoder := gptbot.NewOpenAIEncoder(apiKey, "")
	store := gptbot.NewLocalVectorStore()

	// Feed documents into the vector store.
	feeder := gptbot.NewFeeder(&gptbot.FeederConfig{
		Encoder: encoder,
		Updater: store,
	})
	err := feeder.Feed(ctx, &gptbot.Document{
		ID:   "1",
		Text: "Generative Pre-trained Transformer 3 (GPT-3) is an autoregressive language model released in 2020 that uses deep learning to produce human-like text. Given an initial text as prompt, it will produce text that continues the prompt.\n\nThe architecture is a decoder-only transformer network with a 2048-token-long context and then-unprecedented size of 175 billion parameters, requiring 800GB to store. The model was trained using generative pre-training; it is trained to predict what the next token is based on previous tokens. The model demonstrated strong zero-shot and few-shot learning on many tasks.[2]",
	})
	if err != nil {
		fmt.Printf("err: %v", err)
		return
	}

	// Chat with the bot to get answers.
	bot := gptbot.NewBot(&gptbot.BotConfig{
		APIKey:  apiKey,
		Encoder: encoder,
		Querier: store,
	})

	var history []*gptbot.Turn

	question := "When was GPT-3 released?"
	answer, err := bot.Chat(ctx, question, history...)
	if err != nil {
		fmt.Printf("err: %v", err)
		return
	}
	fmt.Printf("Q: %s\n", question)
	fmt.Printf("A: %s\n", answer)

	// Save the conversation history.
	history = append(history, &gptbot.Turn{
		Question: question,
		Answer:   answer,
	})

	question = "How many parameters does it use?" // In multi-turn mode, here "it" will be inferred to "GPT-3".
	answer, err = bot.Chat(ctx, question, history...)
	if err != nil {
		fmt.Printf("err: %v", err)
		return
	}
	fmt.Printf("Q: %s\n", question)
	fmt.Printf("A: %s\n", answer)

	// Output:
	//
	// Q: When was GPT-3 released?
	// A: GPT-3 was released in 2020.
	// Q: How many parameters does it use?
	// A: GPT-3 uses 175 billion parameters.
}
