package gptbot

import (
	"bytes"
	"context"
	"text/template"

	"github.com/rakyll/openai-go"
	"github.com/rakyll/openai-go/chat"
)

type Embedding []float64

type Section struct {
	Title     string    `json:"title,omitempty"`
	Heading   string    `json:"heading,omitempty"`
	Content   string    `json:"content,omitempty"`
	Embedding Embedding `json:"embedding,omitempty"`
}

type Similarity struct {
	Section

	ID    int     `json:"id,omitempty"`
	Score float64 `json:"score,omitempty"`
}

type Encoder interface {
	Encode(cxt context.Context, text string) (Embedding, error)
	EncodeBatch(cxt context.Context, texts []string) ([]Embedding, error)
}

type Querier interface {
	Query(ctx context.Context, embedding Embedding, topK int) ([]*Similarity, error)
}

type History interface {
	Load(n int) ([]*chat.Message, error)
	Add(q, a *chat.Message) error
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
	// Defaults to 5.
	TopK int

	// History is used to retrieve the history messages if multi-turn conversation is needed.
	// Defaults to `new(LocalHistory)`.
	History History

	// HistoryTurnNum is the number of turns (1/2 of the number of messages) to reserve.
	// Defaults to 0.
	HistoryTurnNum int
}

func (cfg *BotConfig) init() {
	if cfg.Model == "" {
		cfg.Model = "gpt-3.5-turbo"
	}
	if cfg.TopK == 0 {
		cfg.TopK = 5
	}
	if cfg.PromptTmpl == "" {
		cfg.PromptTmpl = DefaultPromptTmpl
	}

	if cfg.History == nil {
		cfg.History = new(LocalHistory)
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

func (b *Bot) Chat(ctx context.Context, question string) (string, error) {
	prompt, err := b.constructPrompt(ctx, question)
	if err != nil {
		return "", err
	}
	questionMessage := &chat.Message{
		Role:    "user",
		Content: prompt,
	}

	var messages []*chat.Message
	historyMessages, err := b.cfg.History.Load(b.cfg.HistoryTurnNum * 2)
	if err != nil {
		return "", err
	}
	messages = append(messages, historyMessages...)
	messages = append(messages, questionMessage)

	resp, err := b.client.CreateCompletion(ctx, &chat.CreateCompletionParams{Messages: messages})
	if err != nil {
		return "", err
	}

	answerMessage := resp.Choices[0].Message
	if err := b.cfg.History.Add(questionMessage, answerMessage); err != nil {
		return "", err
	}

	return answerMessage.Content, nil
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

	var sections []string
	for _, s := range similarities {
		sections = append(sections, s.Content)
	}

	p := PromptTemplate(b.cfg.PromptTmpl)
	return p.Render(PromptData{
		Question: question,
		Sections: sections,
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
