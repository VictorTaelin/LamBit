# LamBit

A minimal, Turing-complete functional language based on binary pattern-matching case-trees.

## Grammar

```
Func ::=
  | Mat ::= "λ" "{" "0" ":" Func ";" "1" ":" Func ";" "}"
  | Lam ::= "λ" Name "." Func
  | Ret ::= Term

Term ::=
  | Var ::= Name
  | Ctr ::= ("0" | "1") "{" [Term ","?] "}"
  | Rec ::= "~" Term
```

A LamBit program is a single top-level function. Computation happens by
recursively pattern-matching that function against a single input via `~`.

## Example

Addition and multiplication on natural numbers:

```
-- Haskell equivalent:
-- add Z     b = b
-- add (S a) b = S (add a b)
-- mul Z     b = Z
-- mul (S a) b = add b (mul a b)

F = λ{
  0: λ{
    0: λb. b
    1: λa. λb. 1{~0{a,b}}
  }
  1: λ{
    0: λb. 0{}
    1: λa. λb. ~0{b,~1{a,b}}
  }
}
```

Where naturals are encoded as `0{}` = 0, `1{0{}}` = 1, `1{1{0{}}}` = 2, etc.

Evaluating `~1{2n,3n}` computes 2*3 = 6:

```
~1{2n,3n}                        -- mul 2 3
= ~0{3n, ~1{1n,3n}}              -- add 3 (mul 1 3)
= ~0{3n, ~0{3n, ~1{0n,3n}}}      -- add 3 (add 3 (mul 0 3))
= ~0{3n, ~0{3n, 0n}}             -- add 3 (add 3 0)
= ~0{3n, 3n}                     -- add 3 3
= 6n                             -- 6
```

## Usage

```bash
npx ts-node lambit.ts
```
