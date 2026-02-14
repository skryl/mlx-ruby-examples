# frozen_string_literal: true

module StableDiffusionExample
  class Tokenizer
    attr_reader :bpe_ranks, :vocab, :max_length

    def initialize(bpe_ranks = nil, vocab = nil, max_length: 77)
      @bpe_ranks = bpe_ranks || {}
      @vocab = vocab || {}
      @max_length = max_length
    end

    def tokenize(text)
      words = text.to_s.downcase.scan(/[a-z0-9']+|[^\s]/)
      ids = [1]
      words.each do |w|
        ids << token_id_for(w)
      end
      ids << 2
      ids = ids.take(max_length)
      ids + Array.new([max_length - ids.length, 0].max, 0)
    end

    def decode(tokens)
      tokens.map(&:to_i).map { |t| vocab.fetch(t.to_s, "tok#{t}") }.join(" ")
    end

    private

    def token_id_for(token)
      if vocab.key?(token)
        vocab[token].to_i
      else
        3 + (token.bytes.sum % 49_000)
      end
    end
  end
end
