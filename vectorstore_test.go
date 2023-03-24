package gptbot_test

import (
	"context"
	"os"
	"testing"

	"github.com/go-aie/gptbot"
	"github.com/google/go-cmp/cmp"
)

func TestLocal_Query(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	encoder := gptbot.NewOpenAIEncoder(apiKey, "")

	store := new(gptbot.LocalVectorStore)
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
					Section: gptbot.Section{
						Title:   "Athletics at the 2020 Summer Olympics – Men's long jump",
						Heading: "Summary",
					},
					ID: 2,
				},
				{
					Section: gptbot.Section{
						Title:   "Athletics at the 2020 Summer Olympics – Men's triple jump",
						Heading: "Summary",
					},
					ID: 3,
				},
				{
					Section: gptbot.Section{
						Title:   "Athletics at the 2020 Summer Olympics – Men's high jump",
						Heading: "Summary",
					},
					ID: 0,
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

		// For simplicity, clear fields Content, Embedding and Score.
		for _, s := range got {
			s.Content = ""
			s.Embedding = nil
			s.Score = 0
		}

		if !cmp.Equal(got, tt.want) {
			diff := cmp.Diff(got, tt.want)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}
