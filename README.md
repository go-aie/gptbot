# GPTBot

[![Go Reference](https://pkg.go.dev/badge/github.com/go-aie/gptbot/vulndb.svg)][1]

Question Answering Bot powered by [OpenAPI GPT models][2].


## Installation

```bash
$ go get -u github.com/go-aie/gptbot
```


## Quick Start

```go
func main() {
    ctx := context.Background()

    store := new(gptbot.LocalVectorStore)
    if err := store.LoadJSON(ctx, "testdata/olympics_sections.json"); err != nil {
        log.Fatalf("err: %v", err)
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
        log.Fatalf("err: %v", err)
    }

    log.Printf("Q: %s", question)
    log.Printf("A: %s", answer)
}
```

Note that the above example uses a loca vector store. If you have a larger dataset, please consider using a vector search engine (e.g. [Milvus](milvus)).


## Design

GPTBot is an implementation of the method demonstrated in [Question Answering using Embeddings][3].

![architecture](docs/architecture.png)


## License

[MIT](LICENSE)


[1]: https://pkg.go.dev/github.com/go-aie/gptbot
[2]: https://platform.openai.com/docs/models
[3]: https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb
