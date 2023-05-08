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

		got, err := store.Query(context.Background(), embedding, "", 3)
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

func TestLocalVectorStore_LoadJSON(t *testing.T) {
	filename, cleanup := createTemp(t, []byte(`[{"id":"id_1","text":"text_1","document_id":"doc_id_1"}]`))
	defer cleanup()

	store := gptbot.NewLocalVectorStore()
	if err := store.LoadJSON(context.Background(), filename); err != nil {
		t.Fatalf("err: %v\n", err)
	}

	got := store.GetAllData(context.Background())
	want := map[string][]*gptbot.Chunk{
		"doc_id_1": {
			{
				ID:         "id_1",
				Text:       "text_1",
				DocumentID: "doc_id_1",
			},
		},
	}

	if !cmp.Equal(got, want) {
		diff := cmp.Diff(got, want)
		t.Errorf("Want - Got: %s", diff)
	}
}

func TestLocalVectorStore_StoreJSON(t *testing.T) {
	store := gptbot.NewLocalVectorStore()
	_ = store.Insert(context.Background(), map[string][]*gptbot.Chunk{
		"doc_id_1": {
			{
				ID:         "id_1",
				Text:       "text_1",
				DocumentID: "doc_id_1",
			},
		},
	})

	filename, cleanup := createTemp(t, []byte(""))
	defer cleanup()

	if err := store.StoreJSON(filename); err != nil {
		t.Fatalf("err: %v\n", err)
	}

	got, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("err: %v\n", err)
	}

	want := []byte(`[{"id":"id_1","text":"text_1","document_id":"doc_id_1","metadata":{}}]`)

	if !cmp.Equal(got, want) {
		diff := cmp.Diff(got, want)
		t.Errorf("Want - Got: %s", diff)
	}
}

func createTemp(t *testing.T, content []byte) (string, func()) {
	f, err := os.CreateTemp("", "test")
	if err != nil {
		t.Fatalf("err: %v\n", err)
	}

	filename := f.Name()
	cleanup := func() {
		_ = os.Remove(filename)
	}

	if _, err := f.Write(content); err != nil {
		cleanup()
		t.Fatalf("err: %v\n", err)
	}
	if err := f.Close(); err != nil {
		cleanup()
		t.Fatalf("err: %v\n", err)
	}

	return filename, cleanup
}
