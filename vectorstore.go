package gptbot

import (
	"context"
	"encoding/json"
	"os"

	"golang.org/x/exp/slices"
	"gonum.org/v1/gonum/mat"
)

type LocalVectorStore struct {
	chunks map[string][]*Chunk
}

func NewLocalVectorStore() *LocalVectorStore {
	return &LocalVectorStore{
		chunks: make(map[string][]*Chunk),
	}
}

func (vs *LocalVectorStore) LoadJSON(ctx context.Context, filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	var chunks []*Chunk
	if err := json.Unmarshal(data, &chunks); err != nil {
		return err
	}

	chunkMap := make(map[string][]*Chunk)
	for _, chunk := range chunks {
		chunkMap[chunk.DocumentID] = append(chunkMap[chunk.DocumentID], chunk)
	}

	return vs.Upsert(ctx, chunkMap)
}

func (vs *LocalVectorStore) Upsert(ctx context.Context, chunks map[string][]*Chunk) error {
	for documentID, chunkList := range chunks {
		vs.chunks[documentID] = chunkList
	}
	return nil
}

func (vs *LocalVectorStore) Query(ctx context.Context, embedding Embedding, topK int) ([]*Similarity, error) {
	if topK <= 0 {
		return nil, nil
	}

	target := mat.NewVecDense(len(embedding), embedding)

	var similarities []*Similarity
	for _, chunks := range vs.chunks {
		for _, chunk := range chunks {
			candidate := mat.NewVecDense(len(chunk.Embedding), chunk.Embedding)
			score := mat.Dot(target, candidate)
			similarities = append(similarities, &Similarity{
				Chunk: chunk,
				Score: score,
			})
		}
	}

	// Sort similarities by score in descending order.
	slices.SortStableFunc(similarities, func(a, b *Similarity) bool {
		return a.Score > b.Score
	})

	if len(similarities) <= topK {
		return similarities, nil
	}
	return similarities[:topK], nil
}

func (vs *LocalVectorStore) Delete(ctx context.Context, documentIDs ...string) error {
	for _, documentID := range documentIDs {
		delete(vs.chunks, documentID)
	}
	return nil
}
