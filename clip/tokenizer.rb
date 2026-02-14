# frozen_string_literal: true

module ClipExample
  class SimpleTokenizer
    attr_reader :pad_id, :bos_id, :eos_id, :max_length

    def initialize(max_length: 77)
      @pad_id = 0
      @bos_id = 1
      @eos_id = 2
      @max_length = max_length
    end

    def vocab_size
      259
    end

    def encode(text)
      ids = [@bos_id]
      ids.concat(text.to_s.bytes.map { |b| b + 3 })
      ids << @eos_id
      ids = ids[0...@max_length]
      if ids.length < @max_length
        ids.concat(Array.new(@max_length - ids.length, @pad_id))
      end
      ids
    end

    def batch_encode(texts)
      texts.map { |t| encode(t) }
    end

    def decode(ids)
      ids.map do |id|
        int_id = id.to_i
        if int_id <= @eos_id
          ""
        else
          (int_id - 3).chr
        end
      end.join
    end
  end
end
