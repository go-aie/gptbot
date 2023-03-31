package gptbot

import (
	"bytes"
	"context"
	"text/template"

	"github.com/rakyll/openai-go"
	"github.com/rakyll/openai-go/chat"
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
	Model string

	// PromptTmpl specifies a custom prompt template.
	// Defaults to DefaultPromptTmpl.
	PromptTmpl string

	// TopK specifies how many candidate similarities will be selected to construct the prompt.
	// Defaults to 3.
	TopK int
}

func (cfg *BotConfig) init() {
	if cfg.Model == "" {
		cfg.Model = "gpt-3.5-turbo"
	}
	if cfg.TopK == 0 {
		cfg.TopK = 3
	}
	if cfg.PromptTmpl == "" {
		cfg.PromptTmpl = DefaultPromptTmpl
	}
}

// Bot is a chatbot powered by OpenAI's GPT models.
type Bot struct {
	client *chat.Client
	cfg    *BotConfig
}

func NewBot(cfg *BotConfig) *Bot {
	cfg.init()
	s := openai.NewSession(cfg.APIKey)

	return &Bot{
		client: chat.NewClient(s, cfg.Model),
		cfg:    cfg,
	}
}

type Conversation interface {
	Messages() []Turn
	Store()
}

func (b *Bot) Chat(ctx context.Context, question string, history ...*Turn) (string, error) {
	prompt, err := b.constructPrompt(ctx, question)
	if err != nil {
		return "", err
	}

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

	resp, err := b.client.CreateCompletion(ctx, &chat.CreateCompletionParams{Messages: messages})
	if err != nil {
		return "", err
	}

	answer := resp.Choices[0].Message.Content
	return answer, nil
}

func (b *Bot) constructPrompt(ctx context.Context, question string) (string, error) {
	emb, err := b.cfg.Encoder.Encode(ctx, question)
	if err != nil {
		return "", err
	}

	similarities, err := b.cfg.Querier.Query(ctx, emb, b.cfg.TopK)
	if err != nil {
		return "", err
	}

	var texts []string
	for _, s := range similarities {
		texts = append(texts, s.Text)
	}

	p := PromptTemplate(b.cfg.PromptTmpl)
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
