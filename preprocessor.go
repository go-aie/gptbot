package gptbot

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/go-aie/xslices"
	"github.com/google/uuid"
	tokenizer "github.com/samber/go-gpt-3-encoder"
)

type PreprocessorConfig struct {
	// ChunkTokenNum is the number of tokens for each text chunk.
	// Defaults to 200.
	ChunkTokenNum int

	// MinChunkCharNum is the minimum number of characters for each text chunk.
	// Defaults to 350.
	MinChunkCharNum int

	// MinChunkLenToEmbed is the minimum length in characters.
	// Chunks with shorter length will be discarded.
	//
	// Defaults to 5.
	MinChunkLenToEmbed int

	// MaxChunkNum is the maximum number of chunks to generate from a text.
	// Defaults to 10000.
	MaxChunkNum int

	// PunctuationMarks is the sentence separators.
	// Defaults to []rune{'.', '?', '!', '。', '？', '！', '\n'}
	PunctuationMarks []rune
}

func (cfg *PreprocessorConfig) init() *PreprocessorConfig {
	if cfg.ChunkTokenNum == 0 {
		cfg.ChunkTokenNum = 200
	}
	if cfg.MinChunkCharNum == 0 {
		cfg.MinChunkCharNum = 350
	}
	if cfg.MinChunkLenToEmbed == 0 {
		cfg.MinChunkLenToEmbed = 5
	}
	if cfg.MaxChunkNum == 0 {
		cfg.MaxChunkNum = 10000
	}
	if len(cfg.PunctuationMarks) == 0 {
		cfg.PunctuationMarks = []rune{'.', '?', '!', '。', '？', '！', '\n'}
	}
	return cfg
}

// Preprocessor splits a list of documents into chunks.
type Preprocessor struct {
	encoder *dummyTokenizer
	cfg     *PreprocessorConfig
}

func NewPreprocessor(cfg *PreprocessorConfig) *Preprocessor {
	return &Preprocessor{
		encoder: newDummyTokenizer(),
		cfg:     cfg.init(),
	}
}

func (p *Preprocessor) Preprocess(docs ...*Document) (map[string][]*Chunk, error) {
	chunkMap := make(map[string][]*Chunk)

	for _, doc := range docs {
		docID := doc.ID
		meta := doc.Metadata
		if docID == "" {
			docID = uuid.New().String()
		}

		textChunks, err := p.split(doc.Text)
		if err != nil {
			return nil, err
		}

		for i, textChunk := range textChunks {
			chunkMap[docID] = append(chunkMap[docID], &Chunk{
				ID:         fmt.Sprintf("%s_%d", docID, i),
				Text:       textChunk,
				DocumentID: docID,
				Metadata:   meta,
			})
		}
	}

	return chunkMap, nil
}

// split converts the text into chunks.
//
// The splitting algorithm is borrowed from https://github.com/openai/chatgpt-retrieval-plugin/blob/88d983585816b7f298edb0cabf7502c5ccff370d/services/chunks.py#L22-L96.
func (p *Preprocessor) split(text string) ([]string, error) {
	if text == "" || strings.TrimSpace(text) == "" {
		return nil, nil
	}

	// Convert the document text into runes.
	runes := []rune(text)

	var chunks []string

	var i int
	var chunkNum int

	for i < len(runes) && chunkNum < p.cfg.MaxChunkNum {
		// Take the first ChunkTokenNum tokens as a chunk.
		chunkRunes, err := p.encoder.Encode(runes[i:], p.cfg.ChunkTokenNum)
		if err != nil {
			return nil, nil
		}

		// Skip the chunk if it is empty or whitespace.
		chunkText := string(chunkRunes)
		if chunkText == "" || strings.TrimSpace(chunkText) == "" {
			i += len(chunkRunes)
			continue
		}

		// Find the last period or punctuation mark in the chunk.
		// Note that here we count the index in runes.
		var lastPuncIdx = -1
		for _, punc := range p.cfg.PunctuationMarks {
			lastPuncIdx = xslices.Max(lastPuncIdx, lastRuneIndex(chunkText, punc))
		}

		if lastPuncIdx != -1 && lastPuncIdx > p.cfg.MinChunkCharNum {
			if chunkRunes[lastPuncIdx] == '.' && lastPuncIdx+1 < len(chunkRunes) {
				// given the dot cases of `equivalent to 66.2 nautical miles` or `http://example.com/download.html`
				// roughly split by: dot mark must followed by space char
				if unicode.IsSpace(chunkRunes[lastPuncIdx+1]) {
					chunkText = string([]rune(chunkText)[:lastPuncIdx+1])
				}
			} else {
				// Truncate the chunk text at the punctuation mark.
				chunkText = string([]rune(chunkText)[:lastPuncIdx+1])
			}
		}

		trimmedChunkText := strings.TrimSpace(strings.ReplaceAll(chunkText, "\n", " "))
		if utf8.RuneCountInString(trimmedChunkText) > p.cfg.MinChunkLenToEmbed {
			chunks = append(chunks, trimmedChunkText)
		}

		i += utf8.RuneCountInString(chunkText)
		chunkNum += 1
	}

	// Handle the remaining runes.
	if i < len(runes) {
		remainingText := string(runes[i:])
		trimmedRemainingText := strings.TrimSpace(strings.ReplaceAll(remainingText, "\n", " "))
		if utf8.RuneCountInString(trimmedRemainingText) > p.cfg.MinChunkLenToEmbed {
			chunks = append(chunks, trimmedRemainingText)
		}
	}

	return chunks, nil
}

func lastRuneIndex(s string, r rune) int {
	runes := []rune(s)
	for i := len(runes) - 1; i >= 0; i-- {
		if runes[i] == r {
			return i
		}
	}
	return -1
}

// dummyTokenizer tokenizes any given string at the rune level, but counts the
// number of tokens as correctly as possible by using go-gpt-3-encoder.
//
// The reason why we do not use go-gpt-3-encoder directly is that it can not
// handle Chinese characters properly.
type dummyTokenizer struct {
	encoder *tokenizer.Encoder
}

func newDummyTokenizer() *dummyTokenizer {
	encoder, err := tokenizer.NewEncoder()
	if err != nil {
		// We assume that there's no error.
		panic(err)
	}
	return &dummyTokenizer{encoder: encoder}
}

// Encode iterates through runes and returns a slice of the leading runes, which
// consume at most tokenNum number of tokens.
func (t *dummyTokenizer) Encode(runes []rune, tokenNum int) ([]rune, error) {
	b := strings.Builder{}
	for i, r := range runes {
		_, _ = b.WriteRune(r)
		tokens, err := t.encoder.Encode(b.String())
		if err != nil {
			return nil, err
		}
		if len(tokens) > tokenNum {
			return runes[:i], nil
		}
	}
	return runes, nil
}
