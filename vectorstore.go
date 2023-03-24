package gptbot

import (
	"context"
	"encoding/json"
	"os"

	"golang.org/x/exp/slices"
	"gonum.org/v1/gonum/mat"
)

type LocalVectorStore struct {
	sections []Section
}

func (vs *LocalVectorStore) LoadJSON(ctx context.Context, filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	var sections []Section
	if err := json.Unmarshal(data, &sections); err != nil {
		return err
	}

	return vs.Insert(ctx, sections)
}

func (vs *LocalVectorStore) Insert(ctx context.Context, sections []Section) error {
	vs.sections = append(vs.sections, sections...)
	return nil
}

func (vs *LocalVectorStore) Query(ctx context.Context, embedding Embedding, topK int) ([]*Similarity, error) {
	if topK <= 0 {
		return nil, nil
	}

	target := mat.NewVecDense(len(embedding), embedding)

	var similarities []*Similarity
	for i, section := range vs.sections {
		candidate := mat.NewVecDense(len(section.Embedding), section.Embedding)
		score := mat.Dot(target, candidate)
		similarities = append(similarities, &Similarity{
			Section: section,
			ID:      i,
			Score:   score,
		})
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
