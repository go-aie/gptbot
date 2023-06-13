package gptbot

import (
	"bytes"
	"context"
	"strings"
	"text/template"

	"github.com/rakyll/openai-go"
)

type ModelType string

const (
	// GPT-4
	GPT4     ModelType = "gpt-4"
	GPT40301 ModelType = "gpt-4-0314"

	// GPT-3.5
	GPT3Dot5Turbo     ModelType = "gpt-3.5-turbo"
	GPT3Dot5Turbo0301 ModelType = "gpt-3.5-turbo-0301"
	TextDavinci003    ModelType = "text-davinci-003"
	TextDavinci002    ModelType = "text-davinci-002"
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
	Query(ctx context.Context, embedding Embedding, corpusID string, topK int) ([]*Similarity, error)
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

	// TopK specifies how many candidate similarities will be selected to construct the prompt.
	// Defaults to 3.
	TopK int

	// Temperature specifies the sampling temperature to use, between 0 and 1.
	// Higher values like 0.8 will make the output more random, while lower values
	// like 0.2 will make it more focused and deterministic. Defaults to 0.7.
	//
	// Note that in multi-turn mode, Temperature only applies to the backend
	// system, and the temperature of the frontend agent is always 0 since we
	// want its behaviour to be as deterministic as possible.
	Temperature float64

	// MaxTokens is the maximum number of tokens to generate in the chat.
	// Defaults to 256.
	MaxTokens int

	// PromptTmpl specifies a custom prompt template for single-turn mode.
	// Defaults to DefaultPromptTmpl.
	PromptTmpl string

	// MultiTurnPromptTmpl specifies a custom prompt template for multi-turn mode.
	// Defaults to DefaultMultiTurnPromptTmpl.
	//
	// Prompt-based question answering bot essentially operates in single-turn mode,
	// since the quality of each answer largely depends on the associated prompt context
	// (i.e. the most similar document chunks), which all depends on the corresponding
	// question rather than the conversation history.
	//
	// As a workaround, we try to achieve the effect of multi-turn mode by adding an
	// extra frontend agent, who can respond directly to the user for casual greetings,
	// and can refine incomplete questions according to the conversation history
	// before consulting the backend system (i.e. the single-turn Question Answering Bot).
	MultiTurnPromptTmpl string
}

func (cfg *BotConfig) init() {
	if cfg.Model == "" {
		cfg.Model = GPT3Dot5Turbo
	}
	if cfg.TopK == 0 {
		cfg.TopK = 3
	}
	if cfg.Temperature == 0 {
		cfg.Temperature = 0.7
	}
	if cfg.MaxTokens == 0 {
		cfg.MaxTokens = 256
	}
	if cfg.PromptTmpl == "" {
		cfg.PromptTmpl = DefaultPromptTmpl
	}
	if cfg.MultiTurnPromptTmpl == "" {
		cfg.MultiTurnPromptTmpl = DefaultMultiTurnPromptTmpl
	}
}

type Bot struct {
	cfg    *BotConfig
	engine Engine
}

// NewBotWithEngine using your specify engine, like Microsoft Azure OpenAI, Baidu Ernie Bot(wenxin yiyan)
func NewBotWithEngine(cfg *BotConfig, engine Engine) *Bot {
	cfg.init()

	return &Bot{
		cfg:    cfg,
		engine: engine,
	}
}

// NewBot using the default engine (openai)
func NewBot(cfg *BotConfig) *Bot {
	cfg.init()
	s := openai.NewSession(cfg.APIKey)
	bot := &Bot{cfg: cfg}

	// https://platform.openai.com/docs/models/model-endpoint-compatibility
	switch cfg.Model {
	case GPT4, GPT40301, GPT3Dot5Turbo, GPT3Dot5Turbo0301:
		bot.engine = NewOpenAIChatEngine(s, cfg.Model)
	case TextDavinci003, TextDavinci002, TextAda001, TextBabbage001, TextCurie001:
		bot.engine = NewOpenAICompletionEngine(s, cfg.Model)
	default:
		panic("unsupported gpt model!")
	}

	return bot
}

// Chat answers the given question in single-turn mode by default. If ChatHistory with non-empty history
// is specified, multi-turn mode will be enabled. See BotConfig.MultiTurnPromptTmpl for more details.
func (b *Bot) Chat(ctx context.Context, question string, options ...ChatOption) (answer string, debug *Debug, err error) {
	opts := new(chatOptions)
	for _, option := range options {
		option(opts)
	}

	if opts.Debug {
		debug = new(Debug)
		ctx = newContext(ctx, debug)
	}

	if len(opts.History) > 0 {
		answer, err = b.multiTurnChat(ctx, question, opts)
		return
	}

	answer, err = b.singleTurnChat(ctx, question, opts)
	return
}

func (b *Bot) multiTurnChat(ctx context.Context, question string, opts *chatOptions) (string, error) {
	prefix := "QUERY:"

	t := PromptTemplate(b.cfg.MultiTurnPromptTmpl)
	prompt, err := t.Render(struct {
		Turns    []*Turn
		Question string
		Prefix   string
	}{
		Turns:    opts.History,
		Question: question,
		Prefix:   prefix,
	})
	if err != nil {
		return "", err
	}

	// Here we set temperature to 0 since we want the output to be focused and deterministic.
	refinedQuestionOrReply, err := b.chat(ctx, prompt, 0)
	if err != nil {
		return "", err
	}

	// Save the reply of the frontend agent for debugging purposes.
	if debug, ok := fromContext(ctx); ok {
		debug.FrontendReply = refinedQuestionOrReply
	}

	if strings.HasPrefix(refinedQuestionOrReply, prefix) {
		return b.singleTurnChat(ctx, refinedQuestionOrReply[len(prefix):], opts)
	} else {
		return refinedQuestionOrReply, nil
	}
}

func (b *Bot) singleTurnChat(ctx context.Context, question string, opts *chatOptions) (string, error) {
	prompt, err := b.cfg.constructPrompt(ctx, question, opts)
	if err != nil {
		return "", err
	}

	// Save the prompt of the backend system for debugging purposes.
	if debug, ok := fromContext(ctx); ok {
		debug.BackendPrompt = prompt
	}

	return b.chat(ctx, prompt, b.cfg.Temperature)
}

func (b *Bot) chat(ctx context.Context, prompt string, temperature float64) (string, error) {
	req := &EngineRequest{
		Messages:    []*EngineMessage{{Role: "user", Content: prompt}},
		Temperature: temperature,
		MaxTokens:   b.cfg.MaxTokens,
	}
	resp, err := b.engine.Infer(ctx, req)
	if err != nil {
		return "", err
	}

	return resp.Text, nil
}

func (b *BotConfig) constructPrompt(ctx context.Context, question string, opts *chatOptions) (string, error) {
	emb, err := b.Encoder.Encode(ctx, question)
	if err != nil {
		return "", err
	}

	similarities, err := b.Querier.Query(ctx, emb, opts.CorpusID, b.TopK)
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

type chatOptions struct {
	Debug    bool
	CorpusID string
	History  []*Turn
}

type ChatOption func(opts *chatOptions)

func ChatDebug(debug bool) ChatOption {
	return func(opts *chatOptions) { opts.Debug = debug }
}

func ChatHistory(history ...*Turn) ChatOption {
	return func(opts *chatOptions) { opts.History = history }
}

func ChatCorpusID(corpusID string) ChatOption {
	return func(opts *chatOptions) { opts.CorpusID = corpusID }
}

type PromptData struct {
	Question string
	Sections []string
}

type PromptTemplate string

func (p PromptTemplate) Render(data any) (string, error) {
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

	DefaultMultiTurnPromptTmpl = `You are an Agent who communicates with the User, with a System available for answering queries. Your responsibilities include:
1. For greetings and pleasantries, respond directly to the User;
2. For other questions, if you cannot understand them, ask the User directly; otherwise, be sure to begin with "{{$.Prefix}}" when querying the System.

Example 1:
User: What is GPT-3?
Agent: {{$.Prefix}} What is GPT-3?

Example 2:
User: How many parameters does it use?
Agent: Sorry, I don't quite understand what you mean.

Example 3:
User: What is GPT-3?
Agent: GPT-3 is an AI model.
User: How many parameters does it use?
Agent: {{$.Prefix}} How many parameters does GPT-3 use?

Conversation:
{{- range $.Turns}}
User: {{.Question}}
Agent: {{.Answer}}
{{- end}}
User: {{$.Question}}
Agent:
`
)

type Debug struct {
	FrontendReply string `json:"frontend_reply,omitempty"`
	BackendPrompt string `json:"backend_prompt,omitempty"`
}

type contextKeyT string

var contextKey = contextKeyT("github.com/go-aie/gptbot/bot.Debug")

// NewContext returns a copy of the parent context
// and associates it with a Debug.
func newContext(ctx context.Context, d *Debug) context.Context {
	return context.WithValue(ctx, contextKey, d)
}

// FromContext returns the Debug bound to the context, if any.
func fromContext(ctx context.Context) (d *Debug, ok bool) {
	d, ok = ctx.Value(contextKey).(*Debug)
	return
}
