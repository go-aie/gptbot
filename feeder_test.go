package gptbot_test

import (
	"context"
	"os"
	"testing"

	"github.com/go-aie/gptbot"
	"github.com/google/go-cmp/cmp"
)

func TestFeeder_Feed(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	encoder := gptbot.NewOpenAIEncoder(apiKey, "")

	store := gptbot.NewLocalVectorStore()
	f := gptbot.NewFeeder(&gptbot.FeederConfig{
		Encoder: encoder,
		Updater: store,
		Preprocessor: gptbot.NewPreprocessor(&gptbot.PreprocessorConfig{
			ChunkTokenNum:   150,
			MinChunkCharNum: 50,
		}),
		BatchSize: 2,
	})

	tests := []struct {
		in   []*gptbot.Document
		want map[string][]*gptbot.Chunk
	}{
		{
			in: []*gptbot.Document{
				{
					ID:   "1",
					Text: "Generative Pre-trained Transformer 3 (GPT-3) is an autoregressive language model released in 2020 that uses deep learning to produce human-like text. Given an initial text as prompt, it will produce text that continues the prompt.\n\nThe architecture is a decoder-only transformer network with a 2048-token-long context and then-unprecedented size of 175 billion parameters, requiring 800GB to store. The model was trained using generative pre-training; it is trained to predict what the next token is based on previous tokens. The model demonstrated strong zero-shot and few-shot learning on many tasks.[2]",
				},
				{
					ID:   "2",
					Text: "生成型预训练变换模型 3 （英语：Generative Pre-trained Transformer 3，简称 GPT-3）是一个自回归语言模型，目的是为了使用深度学习生成人类可以理解的自然语言[1]。GPT-3是由在旧金山的人工智能公司OpenAI训练与开发，模型设计基于谷歌开发的 Transformer 语言模型。GPT-3的神经网络包含1750亿个参数，需要800GB来存储, 为有史以来参数最多的神经网络模型[2]。该模型在许多任务上展示了强大的零样本和少样本的能力。[3]",
				},
			},
			want: map[string][]*gptbot.Chunk{
				"1": {
					{
						ID:         "1_0",
						Text:       "Generative Pre-trained Transformer 3 (GPT-3) is an autoregressive language model released in 2020 that uses deep learning to produce human-like text. Given an initial text as prompt, it will produce text that continues the prompt.  The architecture is a decoder-only transformer network with a 2048-token-long context and then-unprecedented size of 175 billion parameters, requiring 800GB to store. The model was trained using generative pre-training; it is trained to predict what the next token is based on previous tokens. The model demonstrated strong zero-shot and few-shot learning on many tasks.",
						DocumentID: "1",
					},
				},
				"2": {
					{
						ID:         "2_0",
						Text:       "生成型预训练变换模型 3 （英语：Generative Pre-trained Transformer 3，简称 GPT-3）是一个自回归语言模型，目的是为了使用深度学习生成人类可以理解的自然语言[1]。",
						DocumentID: "2",
					},
					{
						ID:         "2_1",
						Text:       "GPT-3是由在旧金山的人工智能公司OpenAI训练与开发，模型设计基于谷歌开发的 Transformer 语言模型。",
						DocumentID: "2",
					},
					{
						ID:         "2_2",
						Text:       "GPT-3的神经网络包含1750亿个参数，需要800GB来存储, 为有史以来参数最多的神经网络模型[2]。该模型在许多任务上展示了强大的零样本和少样本的能力。",
						DocumentID: "2",
					},
				},
			},
		},
	}
	for _, tt := range tests {
		if err := f.Feed(context.Background(), tt.in...); err != nil {
			t.Errorf("err: %v\n", err)
		}

		got := store.GetAllData(context.Background())

		// For simplicity, clear the Embedding field.
		for _, cs := range got {
			for _, c := range cs {
				c.Embedding = nil
			}
		}
		if !cmp.Equal(got, tt.want) {
			diff := cmp.Diff(got, tt.want)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}
