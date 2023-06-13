package gptbot

import (
	"context"
	"log"

	"github.com/rakyll/openai-go"
	"github.com/rakyll/openai-go/chat"
	"github.com/rakyll/openai-go/completion"
)

type EngineMessage struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type EngineRequest struct {
	Messages    []*EngineMessage `json:"messages,omitempty"`
	Temperature float64          `json:"temperature,omitempty"`
	MaxTokens   int              `json:"max_tokens,omitempty"`
}

type EngineResponse struct {
	Text string `json:"text,omitempty"`
}

type Engine interface {
	Infer(context.Context, *EngineRequest) (*EngineResponse, error)
}

// OpenAIChatEngine powered by /v1/chat/completions completion api, supported model like `gpt-4`, `gpt-3.5-turbo` ...
type OpenAIChatEngine struct {
	Client *chat.Client
}

// OpenAICompletionEngine powered by /v1/completions completion api, supported model like `text-davinci-003` ...
type OpenAICompletionEngine struct {
	Client *completion.Client
}

func NewOpenAIChatEngine(s *openai.Session, model ModelType) *OpenAIChatEngine {
	client := chat.NewClient(s, string(model))
	if client == nil {
		log.Fatalf("init open ai chat client error")
	}

	return &OpenAIChatEngine{
		Client: client,
	}
}

func NewOpenAICompletionEngine(s *openai.Session, model ModelType) *OpenAICompletionEngine {
	compClient := completion.NewClient(s, string(model))
	if compClient == nil {
		log.Fatalf("init open ai completion client error")
	}

	return &OpenAICompletionEngine{
		Client: compClient,
	}
}

func (e *OpenAIChatEngine) Infer(ctx context.Context, req *EngineRequest) (resp *EngineResponse, err error) {
	apiResp, err := e.Client.CreateCompletion(ctx, &chat.CreateCompletionParams{
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
		return
	}

	return &EngineResponse{
		Text: apiResp.Choices[0].Message.Content,
	}, nil
}

func (e *OpenAICompletionEngine) Infer(ctx context.Context, req *EngineRequest) (resp *EngineResponse, err error) {
	apiResp, err := e.Client.Create(ctx, &completion.CreateParams{
		Prompt:      []string{req.Messages[0].Content},
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	})
	if err != nil {
		return
	}

	return &EngineResponse{
		Text: apiResp.Choices[0].Text,
	}, nil
}
