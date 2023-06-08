package gptbot

import (
	"context"
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

type OpenAIChatEngine struct {
	// TODO: Implement this engine.
	// gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
}

type OpenAICompletionEngine struct {
	// TODO: Implement this engine.
	// text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001
}
