package gptbot

import (
	"context"

	"github.com/rakyll/openai-go"
	"github.com/rakyll/openai-go/embedding"
)

type OpenAIEncoder struct {
	client *embedding.Client
}

func NewOpenAIEncoder(apiKey string, model string) *OpenAIEncoder {
	if model == "" {
		model = "text-embedding-ada-002"
	}
	s := openai.NewSession(apiKey)

	return &OpenAIEncoder{
		client: embedding.NewClient(s, model),
	}
}

func (e *OpenAIEncoder) Encode(ctx context.Context, text string) (Embedding, error) {
	embeddings, err := e.EncodeBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return embeddings[0], nil
}

func (e *OpenAIEncoder) EncodeBatch(ctx context.Context, texts []string) ([]Embedding, error) {
	resp, err := e.client.Create(ctx, &embedding.CreateParams{
		Input: texts,
	})
	if err != nil {
		return nil, err
	}

	var embeddings []Embedding
	for _, data := range resp.Data {
		embeddings = append(embeddings, data.Embedding)
	}
	return embeddings, nil
}
