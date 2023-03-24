package gptbot_test

import (
	"context"
	"fmt"
	"os"

	"github.com/go-aie/gptbot"
)

func Example() {
	ctx := context.Background()

	store := new(gptbot.LocalVectorStore)
	if err := store.LoadJSON(ctx, "testdata/olympics_sections.json"); err != nil {
		fmt.Printf("err: %v\n", err)
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	bot := gptbot.NewBot(&gptbot.BotConfig{
		APIKey:  apiKey,
		Encoder: gptbot.NewOpenAIEncoder(apiKey, ""),
		Querier: store,
	})

	question := "Who won the 2020 Summer Olympics men's high jump?"
	answer, err := bot.Chat(ctx, question)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}

	fmt.Printf("Q: %s\n", question)
	fmt.Printf("A: %s\n", answer)
}
