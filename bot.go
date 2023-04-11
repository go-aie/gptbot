package gptbot

import (
	"bytes"
	"context"
	"text/template"

	"github.com/rakyll/openai-go"
	"github.com/rakyll/openai-go/chat"
	"github.com/rakyll/openai-go/completion"
)

type ModelType string

const (
	// GPT-4
	GPT4 ModelType = "gpt-4"
	// GPT-3.5
	GPT3Dot5Turbo  ModelType = "gpt-3.5-turbo"
	TextDavinci003 ModelType = "text-davinci-003"
	TextDavinci002 ModelType = "text-davinci-002"
	// GPT-3
	TextAda001     ModelType = "text-ada-001"
	TextCurie001   ModelType = "text-curie-001"
	TextBabbage001 ModelType = "text-babbage-001"
)

type Encoder interface {
	Encode(cxt context.Context, text string) (Embedding, error)
	EncodeBatch(cxt context.Context, texts []string) ([]Embedding, error)
}

type Querier interface {
	Query(ctx context.Context, embedding Embedding, topK int) ([]*Similarity, error)
}

// Turn represents a round of dialogue.
type Turn struct {
	Question string `json:"question,omitempty"`
	Answer   string `json:"answer,omitempty"`
}

type BotConfig struct {
	// APIKey is the OpenAI's APIKey.
	// This field is required.
	APIKey string

	// Encoder is an Embedding Encoder, which will encode the user's question into a vector for similarity search.
	// This field is required.
	Encoder Encoder

	// Querier is a search engine, which is capable of doing the similarity search.
	// This field is required.
	Querier Querier

	// Model is the ID of OpenAI's model to use for chat.
	// Defaults to "gpt-3.5-turbo".
	Model ModelType

	// PromptTmpl specifies a custom prompt template.
	// Defaults to DefaultPromptTmpl.
	PromptTmpl string

	// TopK specifies how many candidate similarities will be selected to construct the prompt.
	// Defaults to 3.
	TopK int

	// The maximum number of tokens to generate in the completion
	// Defaults to 256
	MaxTokens int
}

func (cfg *BotConfig) init() {
	if cfg.Model == "" {
		cfg.Model = GPT3Dot5Turbo
	}
	if cfg.TopK == 0 {
		cfg.TopK = 3
	}
	if cfg.MaxTokens == 0 {
		cfg.MaxTokens = 256
	}
	if cfg.PromptTmpl == "" {
		cfg.PromptTmpl = DefaultPromptTmpl
	}
}

type Bot struct {
	cfg        *BotConfig
	chatClient *chat.Client
	compClient *completion.Client
}

func NewBot(cfg *BotConfig) *Bot {
	cfg.init()
	s := openai.NewSession(cfg.APIKey)
	bot := &Bot{cfg: cfg}

	// https://platform.openai.com/docs/models/model-endpoint-compatibility
	switch cfg.Model {
	case GPT4, GPT3Dot5Turbo:
		bot.chatClient = chat.NewClient(s, string(cfg.Model))
	case TextDavinci003, TextDavinci002, TextAda001, TextBabbage001, TextCurie001:
		bot.compClient = completion.NewClient(s, string(cfg.Model))
	default:
		panic("unsupported gpt model!")
	}

	return bot
}

type Conversation interface {
	Messages() []Turn
	Store()
}

func (b *Bot) Chat(ctx context.Context, question string, history ...*Turn) (string, error) {
	prompt, err := b.cfg.constructPrompt(ctx, question)
	if err != nil {
		return "", err
	}

	if b.chatClient != nil {
		return b.doChatCompletion(ctx, question, prompt, history...)
	}

	return b.doCompletion(ctx, question, prompt, history...)
}

// powered by /v1/chat/completions completion api, supported model like `gpt-3.5-turbo`
func (b *Bot) doChatCompletion(ctx context.Context, question, prompt string, history ...*Turn) (string, error) {
	var messages []*chat.Message
	for _, h := range history {
		messages = append(messages, &chat.Message{
			Role:    "user",
			Content: h.Question,
		})
		messages = append(messages, &chat.Message{
			Role:    "assistant",
			Content: h.Answer,
		})
	}
	messages = append(messages, &chat.Message{
		Role:    "user",
		Content: prompt,
	})

	resp, err := b.chatClient.CreateCompletion(ctx, &chat.CreateCompletionParams{
		MaxTokens: b.cfg.MaxTokens,
		Messages:  messages,
	})
	if err != nil {
		return "", err
	}

	answer := resp.Choices[0].Message.Content
	return answer, nil
}

// powered by /v1/completions completion api, supported model like `text-davinci-003`
func (b *Bot) doCompletion(ctx context.Context, question, prompt string, history ...*Turn) (string, error) {
	resp, err := b.compClient.Create(ctx, &completion.CreateParams{
		MaxTokens: b.cfg.MaxTokens,
		Prompt:    []string{prompt},
	})
	if err != nil {
		return "", err
	}

	answer := resp.Choices[0].Text

	return answer, nil
}

func (b *BotConfig) constructPrompt(ctx context.Context, question string) (string, error) {
	emb, err := b.Encoder.Encode(ctx, question)
	if err != nil {
		return "", err
	}

	similarities, err := b.Querier.Query(ctx, emb, b.TopK)
	if err != nil {
		return "", err
	}

	var texts []string
	for _, s := range similarities {
		texts = append(texts, s.Text)
	}

	p := PromptTemplate(b.PromptTmpl)
	return p.Render(PromptData{
		Question: question,
		Sections: texts,
	})
}

type PromptData struct {
	Question string
	Sections []string
}

type PromptTemplate string

func (p PromptTemplate) Render(data PromptData) (string, error) {
	tmpl, err := template.New("").Parse(string(p))
	if err != nil {
		return "", err
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", err
	}

	return buf.String(), nil
}

const (
	DefaultPromptTmpl = `
Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."

Context:

{{range .Sections -}}
* {{.}}
{{- end}}

Q: {{.Question}}
A:
`
)
