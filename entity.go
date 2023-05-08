package gptbot

type Metadata struct {
	CorpusID string `json:"corpus_id,omitempty"`
}

type Document struct {
	ID       string   `json:"id,omitempty"`
	Text     string   `json:"text,omitempty"`
	Metadata Metadata `json:"metadata,omitempty"`
}

type Embedding []float64

type Chunk struct {
	ID         string    `json:"id,omitempty"`
	Text       string    `json:"text,omitempty"`
	DocumentID string    `json:"document_id,omitempty"`
	Metadata   Metadata  `json:"metadata,omitempty"`
	Embedding  Embedding `json:"embedding,omitempty"`
}

type Similarity struct {
	*Chunk

	Score float64 `json:"score,omitempty"`
}
