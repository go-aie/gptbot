package main

import (
	"context"
	"io"

	"github.com/RussellLuo/kun/pkg/httpcodec"
	"github.com/go-aie/gptbot"
	"github.com/go-aie/gptbot/milvus"
	"github.com/google/uuid"
)

//go:generate kungen ./service.go Service

// Hi here! This is the API documentation for GPTBot.
//
//kun:oas title=GPTBot-API
//kun:oas version=1.0.0
type Service interface {
	// CreateDocuments feeds documents into the vector store.
	//kun:op POST /upsert
	CreateDocuments(ctx context.Context, documents []*gptbot.Document) error

	// UploadFile uploads a file and then feeds the text into the vector store.
	//kun:op POST /upload
	UploadFile(ctx context.Context, file *httpcodec.FormFile) (err error)

	// DeleteDocuments deletes the specified documents from the vector store.
	//kun:op POST /delete
	DeleteDocuments(ctx context.Context, documentIds []string) error

	// Chat sends question to the bot for an answer.
	//kun:op POST /chat
	Chat(ctx context.Context, question string) (answer string, err error)
}

type GPTBot struct {
	feeder *gptbot.Feeder
	store  *milvus.Milvus
	bot    *gptbot.Bot
}

func NewGPTBot(feeder *gptbot.Feeder, store *milvus.Milvus, bot *gptbot.Bot) *GPTBot {
	return &GPTBot{
		feeder: feeder,
		store:  store,
		bot:    bot,
	}
}

func (b *GPTBot) CreateDocuments(ctx context.Context, docs []*gptbot.Document) error {
	return b.feeder.Feed(ctx, docs...)
}

func (b *GPTBot) UploadFile(ctx context.Context, file *httpcodec.FormFile) (err error) {
	defer file.File.Close()
	data, err := io.ReadAll(file.File)
	if err != nil {
		return err
	}

	doc := &gptbot.Document{
		ID:   uuid.New().String(),
		Text: string(data),
	}
	return b.feeder.Feed(ctx, doc)
}

func (b *GPTBot) DeleteDocuments(ctx context.Context, docIDs []string) error {
	return b.store.Delete(ctx, docIDs...)
}

func (b *GPTBot) Chat(ctx context.Context, question string) (answer string, err error) {
	return b.bot.Chat(ctx, question)
}
