package gptbot_test

import (
	"context"
	"os"
	"testing"

	"github.com/go-aie/gptbot"
	"github.com/google/go-cmp/cmp"
)

func TestLocalVectorStore_Query(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	encoder := gptbot.NewOpenAIEncoder(apiKey, "")

	store := gptbot.NewLocalVectorStore()
	if err := store.LoadJSON(context.Background(), "testdata/olympics_sections.json"); err != nil {
		t.Fatalf("err: %v\n", err)
	}

	tests := []struct {
		in   string
		want []*gptbot.Similarity
	}{
		{
			in: "Who won the 2020 Summer Olympics men's high jump?",
			want: []*gptbot.Similarity{
				{
					Chunk: &gptbot.Chunk{
						ID:         "Men's long jump - Summary",
						DocumentID: "Athletics at the 2020 Summer Olympics",
					},
				},
				{
					Chunk: &gptbot.Chunk{
						ID:         "Men's triple jump - Summary",
						DocumentID: "Athletics at the 2020 Summer Olympics",
					},
				},
				{
					Chunk: &gptbot.Chunk{
						ID:         "Men's high jump - Summary",
						DocumentID: "Athletics at the 2020 Summer Olympics",
					},
				},
			},
		},
	}
	for _, tt := range tests {
		embedding, err := encoder.Encode(context.Background(), tt.in)
		if err != nil {
			t.Errorf("err: %v\n", err)
		}

		got, err := store.Query(context.Background(), embedding, 3)
		if err != nil {
			t.Errorf("err: %v\n", err)
		}

		// For simplicity, clear fields Text, Embedding and Score.
		for _, s := range got {
			s.Text = ""
			s.Embedding = nil
			s.Score = 0
		}

		if !cmp.Equal(got, tt.want) {
			diff := cmp.Diff(got, tt.want)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}
