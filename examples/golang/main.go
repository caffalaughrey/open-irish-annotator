package main

import (
    "encoding/json"
    "flag"
    "fmt"
    "log"
    "os"
    "path/filepath"

    ort "github.com/microsoft/onnxruntime-go"
)

func mustReadJSON(path string, v any) {
    b, err := os.ReadFile(path)
    if err != nil { log.Fatalf("read %s: %v", path, err) }
    if err := json.Unmarshal(b, v); err != nil { log.Fatalf("json %s: %v", path, err) }
}

func main() {
    modelPath := flag.String("model", "artifacts/onnx/model.onnx", "path to model.onnx")
    resDir := flag.String("resources", "rust/morphology_runtime/resources", "resources dir")
    preferLex := flag.Bool("prefer-lexicon", false, "prefer lemma_lexicon.json if present")
    flag.Parse()
    tokens := flag.Args()
    if len(tokens) == 0 {
        log.Fatalf("usage: go run ./examples/golang -- [--model path] [--resources dir] <token> ...")
    }

    var tag2id map[string]int
    var word2id map[string]int
    var char2id map[string]int
    mustReadJSON(filepath.Join(*resDir, "tagset.json"), &tag2id)
    mustReadJSON(filepath.Join(*resDir, "word_vocab.json"), &word2id)
    mustReadJSON(filepath.Join(*resDir, "char_vocab.json"), &char2id)
    id2tag := make([]string, len(tag2id))
    for k, v := range tag2id { if v >= 0 && v < len(id2tag) { id2tag[v] = k } }

    lemmaLex := map[string]string{}
    if *preferLex {
        p := filepath.Join(*resDir, "lemma_lexicon.json")
        if _, err := os.Stat(p); err == nil { mustReadJSON(p, &lemmaLex) }
    }

    t := len(tokens)
    maxChars := 24
    wordIds := make([]int64, 1*t)
    charIds := make([]int64, 1*t*maxChars)
    for i, tok := range tokens {
        if id, ok := word2id[tok]; ok { wordIds[i] = int64(id) } else { wordIds[i] = 1 }
        r := []rune(tok)
        for j := 0; j < len(r) && j < maxChars-1; j++ {
            if id, ok := char2id[string(r[j])]; ok { charIds[i*maxChars+j] = int64(id) } else { charIds[i*maxChars+j] = 2 }
        }
    }

    env, err := ort.NewEnvironment()
    if err != nil { log.Fatalf("env: %v", err) }
    defer env.Destroy()
    sessOpts := ort.NewSessionOptions()
    defer sessOpts.Destroy()
    session, err := env.NewSession(*modelPath, sessOpts)
    if err != nil { log.Fatalf("session: %v", err) }
    defer session.Destroy()

    wi, err := ort.NewTensor(ort.TENSOR_INT64, []int64{1, int64(t)}, wordIds)
    if err != nil { log.Fatalf("tensor word: %v", err) }
    defer wi.Destroy()
    ci, err := ort.NewTensor(ort.TENSOR_INT64, []int64{1, int64(t), int64(maxChars)}, charIds)
    if err != nil { log.Fatalf("tensor char: %v", err) }
    defer ci.Destroy()
    outs, err := session.Run(map[string]*ort.Value{"word_ids": &wi, "char_ids": &ci})
    if err != nil { log.Fatalf("run: %v", err) }
    tagLogits := outs[0].Float32s()
    lemmaLogits := outs[1].Float32s()

    K := len(tagLogits) / t
    tagIdx := make([]int, t)
    for i := 0; i < t; i++ {
        maxV := float32(-1e30)
        maxJ := 0
        for j := 0; j < K; j++ {
            v := tagLogits[i*K+j]
            if v > maxV { maxV, maxJ = v, j }
        }
        tagIdx[i] = maxJ
    }

    LxC := len(lemmaLogits) / t
    L := 24
    C := LxC / L

    id2ch := make([]string, len(char2id))
    for k, v := range char2id { if v >= 0 && v < len(id2ch) { id2ch[v] = k } }

    for i, tok := range tokens {
        tag := id2tag[tagIdx[i]]
        start := i * L * C
        decoded := tok
        lemmaChars := make([]rune, 0, L)
        for step := 0; step < L; step++ {
            maxV := float32(-1e30)
            maxC := 0
            base := start + step*C
            for j := 0; j < C; j++ {
                v := lemmaLogits[base+j]
                if v > maxV { maxV, maxC = v, j }
            }
            if maxC == 1 { break }
            ch := '?' 
            if id2ch[maxC] != "" { rs := []rune(id2ch[maxC]); if len(rs)>0 { ch = rs[0] } }
            lemmaChars = append(lemmaChars, ch)
        }
        if len(lemmaChars) > 0 { decoded = string(lemmaChars) }
        if *preferLex { if l, ok := lemmaLex[tok]; ok { decoded = l } }
        fmt.Printf("%s\t%s\t%s\n", tok, tag, decoded)
    }
}
