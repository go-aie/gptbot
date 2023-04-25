package gptbot

import (
	"context"
)

type XPreprocessor interface {
	Preprocess(docs ...*Document) (map[string][]*Chunk, error)
}

type Updater interface {
	Insert(ctx context.Context, chunks map[string][]*Chunk) error
	Delete(ctx context.Context, documentIDs ...string) error
}

type FeederConfig struct {
	// Encoder is the embedding encoder.
	// This field is required.
	Encoder Encoder

	// Updater is the vector store for inserting/deleting chunks.
	// This field is required.
	Updater Updater

	// Defaults to NewPreprocessor(...).
	Preprocessor XPreprocessor

	// BatchSize is the number of chunks to encode/upsert at a time.
	// Defaults to 100.
	BatchSize int
}

func (cfg *FeederConfig) init() *FeederConfig {
	if cfg.Preprocessor == nil {
		cfg.Preprocessor = NewPreprocessor(&PreprocessorConfig{})
	}
	if cfg.BatchSize == 0 {
		cfg.BatchSize = 100
	}
	return cfg
}

type Feeder struct {
	cfg *FeederConfig
}

func NewFeeder(cfg *FeederConfig) *Feeder {
	return &Feeder{
		cfg: cfg.init(),
	}
}

func (f *Feeder) Preprocessor() XPreprocessor {
	return f.cfg.Preprocessor
}

func (f *Feeder) Feed(ctx context.Context, docs ...*Document) error {
	chunks, err := f.cfg.Preprocessor.Preprocess(docs...)
	if err != nil {
		return err
	}

	// Delete old chunks belonging to the given document IDs.
	var docIDs []string
	for docID := range chunks {
		docIDs = append(docIDs, docID)
	}
	if err := f.cfg.Updater.Delete(ctx, docIDs...); err != nil {
		return err
	}

	// Insert new chunks.
	for batch := range genBatches(chunks, f.cfg.BatchSize) {
		if err := f.encode(ctx, batch); err != nil {
			return err
		}
		if err := f.insert(ctx, batch); err != nil {
			return err
		}
	}

	return nil
}

func (f *Feeder) encode(ctx context.Context, batch []*Chunk) error {
	var texts []string
	for _, chunk := range batch {
		texts = append(texts, chunk.Text)
	}

	embeddings, err := f.cfg.Encoder.EncodeBatch(ctx, texts)
	if err != nil {
		return err
	}

	for i, chunk := range batch {
		chunk.Embedding = embeddings[i]
	}

	return nil
}

func (f *Feeder) insert(ctx context.Context, batch []*Chunk) error {
	chunkMap := make(map[string][]*Chunk)
	for _, chunk := range batch {
		chunkMap[chunk.DocumentID] = append(chunkMap[chunk.DocumentID], chunk)
	}
	return f.cfg.Updater.Insert(ctx, chunkMap)
}

func genBatches(chunks map[string][]*Chunk, size int) <-chan []*Chunk {
	ch := make(chan []*Chunk)

	go func() {
		var batch []*Chunk

		for _, chunkList := range chunks {
			for _, chunk := range chunkList {
				batch = append(batch, chunk)

				if len(batch) == size {
					// Reach the batch size, copy and send all the buffered chunks.
					temp := make([]*Chunk, size)
					copy(temp, batch)
					ch <- temp

					// Clear the buffer.
					batch = batch[:0]
				}
			}
		}

		// Send all the remaining chunks, if any.
		if len(batch) > 0 {
			ch <- batch
		}

		close(ch)
	}()

	return ch
}
