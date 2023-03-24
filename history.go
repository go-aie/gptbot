package gptbot

import (
	"github.com/rakyll/openai-go/chat"
)

type LocalHistory struct {
	messages []*chat.Message
}

func (h *LocalHistory) Load(n int) ([]*chat.Message, error) {
	diff := len(h.messages) - n
	if diff <= 0 {
		return h.messages, nil
	}
	return h.messages[diff:], nil
}

func (h *LocalHistory) Add(q, a *chat.Message) error {
	h.messages = append(h.messages, q, a)
	return nil
}
