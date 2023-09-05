package gptbot

import (
	"context"

	"github.com/rakyll/openai-go"
	"github.com/rakyll/openai-go/chat"
	"github.com/rakyll/openai-go/completion"
)

type ModelType string

const (
	// GPT-4
	GPT4        ModelType = "gpt-4"
	GPT40613    ModelType = "gpt-4-0613"
	GPT432K     ModelType = "gpt-4-32k"
	GPT432K0613 ModelType = "gpt-4-32k-0613"
	GPT40314    ModelType = "gpt-4-0314"
	GPT432K0314 ModelType = "gpt-4-32k-0314"

	// GPT-3.5
	GPT3Dot5Turbo        ModelType = "gpt-3.5-turbo"
	GPT3Dot5Turbo16K     ModelType = "gpt-3.5-turbo-16k"
	GPT3Dot5Turbo0613    ModelType = "gpt-3.5-turbo-0613"
	GPT3Dot5Turbo16K0613 ModelType = "gpt-3.5-turbo-16k-0613"
	TextDavinci003       ModelType = "text-davinci-003"
	TextDavinci002       ModelType = "text-davinci-002"

	// GPT-3 (not recommend)
	TextAda001     ModelType = "text-ada-001"
	TextCurie001   ModelType = "text-curie-001"
	TextBabbage001 ModelType = "text-babbage-001"
)

// OpenAIChatEngine is an engine powered by OpenAI's Chat API /v1/chat/completions.
//
// See https://platform.openai.com/docs/models/model-endpoint-compatibility for
// the supported models.
type OpenAIChatEngine struct {
	Client *chat.Client
}

func NewOpenAIChatEngine(apiKey string, model ModelType) *OpenAIChatEngine {
	return &OpenAIChatEngine{
		Client: chat.NewClient(openai.NewSession(apiKey), string(model)),
	}
}

func (e *OpenAIChatEngine) Infer(ctx context.Context, req *EngineRequest) (*EngineResponse, error) {
	resp, err := e.Client.CreateCompletion(ctx, &chat.CreateCompletionParams{
		Messages: []*chat.Message{
			{
				Role:    req.Messages[0].Role,
				Content: req.Messages[0].Content,
			},
		},
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	})
	if err != nil {
		return nil, err
	}

	return &EngineResponse{
		Text: resp.Choices[0].Message.Content,
	}, nil
}

// OpenAICompletionEngine is an engine powered by OpenAI's Completion API /v1/completions.
//
// See https://platform.openai.com/docs/models/model-endpoint-compatibility for
// the supported models.
type OpenAICompletionEngine struct {
	Client *completion.Client
}

func NewOpenAICompletionEngine(apiKey string, model ModelType) *OpenAICompletionEngine {
	return &OpenAICompletionEngine{
		Client: completion.NewClient(openai.NewSession(apiKey), string(model)),
	}
}

func (e *OpenAICompletionEngine) Infer(ctx context.Context, req *EngineRequest) (*EngineResponse, error) {
	resp, err := e.Client.Create(ctx, &completion.CreateParams{
		Prompt:      []string{req.Messages[0].Content},
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	})
	if err != nil {
		return nil, err
	}

	return &EngineResponse{
		Text: resp.Choices[0].Text,
	}, nil
}
