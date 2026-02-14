# frozen_string_literal: true

require "json"
require "open3"
require "open-uri"
require "pathname"
require "tmpdir"

module LoraExample
  module Data
    module WikiSQL
      module_function

      def load
        %w[train dev test].map { |name| Dataset.new(name) }
      end

      class Dataset
        def initialize(dataset, save_dir: "/tmp")
          valid = %w[train dev test]
          unless valid.include?(dataset)
            raise ArgumentError, "Dataset must be one of #{valid.join(', ')}, got #{dataset.inspect}"
          end

          data_dir = Pathname.new(save_dir).join("wikisql")
          maybe_download(data_dir)

          parse_tables(data_dir.join("data", "#{dataset}.tables.jsonl"))
          parse_queries(data_dir.join("data", "#{dataset}.jsonl"))
        end

        def [](idx)
          @queries[idx]
        end

        def length
          @queries.length
        end

        private

        def maybe_download(data_dir)
          return if data_dir.exist?

          url = "https://raw.githubusercontent.com/salesforce/WikiSQL/master/data.tar.bz2"
          data_dir.mkpath
          archive = URI.open(url, &:read)
          Dir.mktmpdir("wikisql-") do |tmpdir|
            tmp_path = Pathname.new(tmpdir).join("data.tar.bz2")
            File.binwrite(tmp_path, archive)
            extract_tar_bz2(tmp_path, data_dir)
          end
        end

        def extract_tar_bz2(path, destination)
          destination.mkpath
          stdout, stderr, status = Open3.capture3(
            "tar",
            "-xjf",
            path.to_s,
            "-C",
            destination.to_s
          )
          return if status.success?

          raise RuntimeError, "Failed to extract #{path}: #{stderr}\n#{stdout}"
        end

        def parse_tables(path)
          @tables = {}
          File.foreach(path) do |line|
            table = JSON.parse(line)
            @tables[table.fetch("id")] = {
              "columns" => table.fetch("header"),
              "types" => table.fetch("types"),
              "desc" => "table: #{table.fetch('id')}\ncolumns: #{table.fetch('header').join(', ')}"
            }
          end
        end

        def parse_queries(path)
          @queries = []
          File.foreach(path) do |line|
            query = JSON.parse(line)
            table = @tables.fetch(query.fetch("table_id"))
            question = query.fetch("question")
            answer = query_to_text(
              query.fetch("sql"),
              query.fetch("table_id"),
              table.fetch("columns"),
              table.fetch("types")
            )
            @queries << "<s>#{table.fetch('desc')}\nQ: #{question}\nA: #{answer}</s>"
          end
        end

        def query_to_text(query, table, columns, types)
          aggregation_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
          condition_ops = ["=", ">", "<", "OP"]
          column = columns[query.fetch("sel")]
          aggregation = query.fetch("agg").positive? ? "#{aggregation_ops[query.fetch('agg')]} " : ""
          sql = "SELECT #{aggregation}#{column} FROM #{table}"

          conditions = query.fetch("conds")
          unless conditions.empty?
            parts = conditions.map do |i, op_idx, value|
              col = columns[i]
              op = condition_ops[op_idx]
              rendered_value = types[i] == "text" ? "'#{value}'" : value
              "#{col} #{op} #{rendered_value}"
            end
            sql += " WHERE #{parts.join(' AND ')}"
          end
          sql
        end
      end
    end
  end
end

if $PROGRAM_NAME == __FILE__
  # Small smoke sanity for local conversion utility.
  puts "WikiSQL utility loaded."
end
