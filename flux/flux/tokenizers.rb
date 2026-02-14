# frozen_string_literal: true

require "json"

module FluxExample
  class CLIPTokenizer
    attr_reader :bpe_ranks, :vocab, :max_length

    def initialize(bpe_ranks = nil, vocab = nil, max_length: 77)
      @bpe_ranks = bpe_ranks || {}
      @vocab = vocab || {}
      @max_length = max_length
      @cache = { bos => [bos], eos => [eos] }
    end

    def bos
      "<|startoftext|>"
    end

    def eos
      "<|endoftext|>"
    end

    def bos_token
      vocab.fetch(bos, 1)
    end

    def eos_token
      vocab.fetch(eos, 2)
    end

    def bpe(text)
      return @cache[text] if @cache.key?(text)

      pieces = text.to_s.chars
      pieces[-1] = "#{pieces[-1]}</w>" unless pieces.empty?
      @cache[text] = pieces
      pieces
    end

    def tokenize(text, prepend_bos: true, append_eos: true)
      if text.is_a?(Array)
        return text.map { |t| tokenize(t, prepend_bos: prepend_bos, append_eos: append_eos) }
      end

      chunks = text.to_s.downcase.scan(/[a-z0-9']+|[^\s]/)
      bpe_tokens = chunks.flat_map { |t| bpe(t) }
      ids = bpe_tokens.map { |t| vocab.fetch(t, 3 + (t.bytes.sum % 49_000)) }
      ids.unshift(bos_token) if prepend_bos
      ids << eos_token if append_eos
      if ids.length > max_length
        ids = ids.take(max_length)
        ids[-1] = eos_token if append_eos
      end
      ids
    end

    def encode(text)
      if !text.is_a?(Array)
        return encode([text])
      end

      tokens = tokenize(text)
      length = tokens.map(&:length).max || 0
      padded = tokens.map do |row|
        row + Array.new(length - row.length, eos_token)
      end
      MLX::Core.array(padded, MLX::Core.int32)
    end
  end

  class T5Tokenizer
    attr_reader :model_file, :max_length

    def initialize(model_file = nil, max_length: 512)
      @model_file = model_file
      @max_length = max_length
    end

    def pad_token
      0
    end

    def bos_token
      1
    end

    def eos_token
      2
    end

    def tokenize(text, prepend_bos: true, append_eos: true, pad: true)
      if text.is_a?(Array)
        return text.map do |t|
          tokenize(t, prepend_bos: prepend_bos, append_eos: append_eos, pad: pad)
        end
      end

      ids = text.to_s.downcase.scan(/[a-z0-9']+|[^\s]/).map { |tok| 3 + (tok.bytes.sum % 32_000) }
      ids.unshift(bos_token) if prepend_bos
      ids << eos_token if append_eos
      ids = ids.take(max_length)
      ids += Array.new(max_length - ids.length, pad_token) if pad && ids.length < max_length
      ids
    end

    def encode(text, pad: true)
      if !text.is_a?(Array)
        return encode([text], pad: pad)
      end

      tokens = tokenize(text, pad: pad)
      length = tokens.map(&:length).max || 0
      padded = tokens.map { |row| row + Array.new(length - row.length, pad_token) }
      MLX::Core.array(padded, MLX::Core.int32)
    end
  end
end
