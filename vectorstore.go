package gptbot

import (
	"context"
	"encoding/json"
	"os"

	"golang.org/x/exp/maps"
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

// LoadJSON will deserialize from disk into a `LocalVectorStore` based on the provided filename.
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

	return vs.Insert(ctx, chunkMap)
}

// StoreJSON will serialize the `LocalVectorStore` to disk based on the provided filename.
func (vs *LocalVectorStore) StoreJSON(filename string) error {
	var chunks []*Chunk

	for _, chunk := range vs.chunks {
		for _, c := range chunk {
			chunks = append(chunks, c)
		}
	}

	b, err := json.Marshal(chunks)
	if err != nil {
		return err
	}

	err = os.WriteFile(filename, b, 0666)
	if err != nil {
		return err
	}

	return nil
}

// GetAllData returns all the internal data. It is mainly used for testing purpose.
func (vs *LocalVectorStore) GetAllData(ctx context.Context) map[string][]*Chunk {
	return vs.chunks
}

func (vs *LocalVectorStore) Insert(ctx context.Context, chunks map[string][]*Chunk) error {
	for documentID, chunkList := range chunks {
		vs.chunks[documentID] = append(vs.chunks[documentID], chunkList...)
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

// Delete deletes the chunks belonging to the given documentIDs.
// As a special case, empty documentIDs means deleting all chunks.
func (vs *LocalVectorStore) Delete(ctx context.Context, documentIDs ...string) error {
	if len(documentIDs) == 0 {
		maps.Clear(vs.chunks)
	}
	for _, documentID := range documentIDs {
		delete(vs.chunks, documentID)
	}
	return nil
}
