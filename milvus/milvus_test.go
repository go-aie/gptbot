package milvus_test

import (
	"context"
	"os"
	"testing"

	"github.com/go-aie/gptbot"
	"github.com/go-aie/gptbot/milvus"
	"github.com/google/go-cmp/cmp"
)

func TestMilvus_Query(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	encoder := gptbot.NewOpenAIEncoder(apiKey, "")

	store, err := milvus.NewMilvus(&milvus.Config{
		CollectionName: "olympics_knowledge",
	})
	if err != nil {
		t.Fatalf("err: %v\n", err)
	}

	if err := store.LoadJSON(context.Background(), "../testdata/olympics_sections.json"); err != nil {
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
						ID:         "Summary",
						DocumentID: "Athletics at the 2020 Summer Olympics - Men's long jump",
					},
				},
				{
					Chunk: &gptbot.Chunk{
						ID:         "Summary",
						DocumentID: "Athletics at the 2020 Summer Olympics - Men's triple jump",
					},
				},
				{
					Chunk: &gptbot.Chunk{
						ID:         "Summary",
						DocumentID: "Athletics at the 2020 Summer Olympics - Men's high jump",
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
