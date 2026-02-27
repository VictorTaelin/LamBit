// LamBit.ts
// =========
// A minimal functional language based on binary pattern-matching case-trees.
//
//     Func ::=
//       | Mat ::= "λ" "{" "0" ":" Func ";" "1" ":" Func ";" "}"
//       | Lam ::= "λ" Name "." Func
//       | Ret ::= Term
//
//     Term ::=
//       | Var ::= Name
//       | Ctr ::= ("0" | "1") "{" [Term ","?] "}"
//       | Rec ::= "~" Term
//
// LamBit programs are composed by a single "top-level" function. Computation
// happens by recursively pattern-matching that function against a single input.
// 
// LamBit is Turing-complete. For example, consider the following Haskell program:
// 
//     data Nat    = Z | S Nat
//     add Z     b = b
//     add (S a) b = S (add a b)
//     mul Z     b = Z
//     mul (S a) b = add b (mul a b)
// 
// It can be expressed on LamBit as:
//
//     F = λ{
//       0: λ{ // computes addition
//         0: λb. b
//         1: λa. λb. 1{~0{a,b}}
//       }
//       1: λ{ // computes multiplication
//         0: λb. 0{}
//         1: λa. λb. ~0{b,~1{a,b}}
//       }
//     }
//
// Where natural numbers are encoded as:
//
//     Nat ::=
//     | zero ::= 0{}
//     | succ ::= 1{Nat}
//     (aliasing 0n as 0{}, 1n as 1{0{}}, 2n as 1{1{0{}}}, etc.)
//
// For instance, ~1{2n,3n} evaluates 2n*3n to 6n:
//
//     ~1{2n,3n}
//     = ~0{3n, ~1{1n,3n}}
//     = ~0{3n, ~0{3n, ~1{0n,3n}}}
//     = ~0{3n, ~0{3n, 0n}}
//     = ~0{3n, 3n}
//     = 1{~0{2n,3n}}
//     = 1{1{~0{1n,3n}}}
//     = 1{1{1{~0{0n,3n}}}}
//     = 1{1{1{3n}}}
//     = 6n

// Types
// -----

// Function-side: patterns and lambdas
type Mat  = { $: "Mat", zro: Func, one: Func };
type Lam  = { $: "Lam", nam: string, bod: (x: Term) => Func };
type Ret  = { $: "Ret", trm: Term };
type Func = Mat | Lam | Ret;

// Data-side: values and expressions
type Var  = { $: "Var", nam: string };
type Ctr  = { $: "Ctr", tag: number, fds: Term[] };
type Rec  = { $: "Rec", arg: Term };
type Term = Var | Ctr | Rec;

// Interaction stats
type Stats = { app_fun: number, app_lam: number, app_mat: number };

// Constructors
const Mat = (zro, one) => ({ $: "Mat", zro, one }) as Func;
const Lam = (nam, bod) => ({ $: "Lam", nam, bod }) as Func;
const Ret = (trm)      => ({ $: "Ret", trm }) as Func;
const Var = (nam)      => ({ $: "Var", nam }) as Term;
const Ctr = (tag, fds) => ({ $: "Ctr", tag, fds }) as Term;
const Rec = (arg)      => ({ $: "Rec", arg }) as Term;

// Creates a fresh stats counter
function new_stats(): Stats {
  return { app_fun: 0, app_lam: 0, app_mat: 0 };
}

// Parser
// ------

type Parser = { src: string, idx: number };

// Skips whitespace
function skip(p: Parser): void {
  while (p.idx < p.src.length && /\s/.test(p.src[p.idx])) {
    p.idx++;
  }
}

// Tries to consume a string, returns success
function match(p: Parser, str: string): boolean {
  skip(p);
  if (p.src.startsWith(str, p.idx)) {
    p.idx += str.length;
    return true;
  }
  return false;
}

// Consumes a string or throws
function expect(p: Parser, str: string): void {
  if (!match(p, str)) {
    throw new Error(`Expected '${str}' at index ${p.idx}`);
  }
}

// Peeks at current char after skipping whitespace
function peek(p: Parser): string {
  skip(p);
  return p.src[p.idx] || "";
}

// Parses an alphanumeric name
function parse_name(p: Parser): string {
  skip(p);
  var start = p.idx;
  while (p.idx < p.src.length && /[a-zA-Z_0-9]/.test(p.src[p.idx])) {
    p.idx++;
  }
  if (p.idx === start) {
    throw new Error(`Expected name at index ${p.idx}`);
  }
  return p.src.slice(start, p.idx);
}

// Checks if a tag digit is followed by '{'
function is_ctr(p: Parser): boolean {
  skip(p);
  var c = p.src[p.idx];
  return (c === "0" || c === "1") && p.src[p.idx + 1] === "{";
}

// Parses a Func with a name context for HOAS
function parse_func(p: Parser, ctx: Map<string, Term>): Func {
  skip(p);
  // Mat or Lam
  if (match(p, "λ")) {
    // Mat: λ{ 0: ... 1: ... }
    if (match(p, "{")) {
      expect(p, "0");
      expect(p, ":");
      var zro = parse_func(p, ctx);
      match(p, ";");
      expect(p, "1");
      expect(p, ":");
      var one = parse_func(p, ctx);
      match(p, ";");
      expect(p, "}");
      return Mat(zro, one);
    }
    // Lam: λName. Func
    var nam = parse_name(p);
    expect(p, ".");
    return parse_func_lam(p, ctx, nam);
  }
  // Ret: Term
  var trm = parse_term(p, ctx);
  return Ret(trm);
}

// Parses a lambda, advancing the parser correctly
function parse_func_lam(p: Parser, ctx: Map<string, Term>, nam: string): Func {
  // Parse body once with sentinel to find end position
  var sentinel   = Var("$" + nam);
  var ctx2       = new Map(ctx);
  ctx2.set(nam, sentinel);
  var body_start = p.idx;
  parse_func(p, ctx2);
  var body_end   = p.idx;
  // Return a Lam that re-parses body with the actual argument
  return Lam(nam, (x: Term) => {
    var ctx3 = new Map(ctx);
    ctx3.set(nam, x);
    var bp = { src: p.src, idx: body_start };
    return parse_func(bp, ctx3);
  });
}

// Parses a Term with a name context
function parse_term(p: Parser, ctx: Map<string, Term>): Term {
  skip(p);
  // Rec: ~Term
  if (match(p, "~")) {
    var arg = parse_term(p, ctx);
    return Rec(arg);
  }
  // Ctr: 0{...} or 1{...}
  if (is_ctr(p)) {
    var tag = p.src[p.idx] === "0" ? 0 : 1;
    p.idx++;
    expect(p, "{");
    var fds: Term[] = [];
    while (!match(p, "}")) {
      if (fds.length > 0) {
        match(p, ",");
      }
      fds.push(parse_term(p, ctx));
    }
    return Ctr(tag, fds);
  }
  // Var: Name
  var nam = parse_name(p);
  var val = ctx.get(nam);
  if (val === undefined) {
    throw new Error(`Unbound variable '${nam}'`);
  }
  return val;
}

// Evaluation
// ----------

// Feeds a single term into a Func (handles Lam and Mat)
function feed(prog: Func, func: Func, term: Term, stats: Stats): Func {
  switch (func.$) {
    case "Lam": {
      stats.app_lam++;
      return func.bod(term);
    }
    case "Mat": {
      if (term.$ !== "Ctr") {
        throw new Error("Mat expected a Ctr");
      }
      stats.app_mat++;
      var branch = term.tag === 0 ? func.zro : func.one;
      var result = branch as Func;
      for (var i = 0; i < term.fds.length; i++) {
        result = feed(prog, result, term.fds[i], stats);
      }
      return result;
    }
    default: {
      throw new Error("Cannot feed into a Ret");
    }
  }
}

// Reduces a term to normal form
function eval_term(prog: Func, term: Term, stats: Stats): Term {
  switch (term.$) {
    case "Var": {
      return term;
    }
    case "Ctr": {
      var fds = term.fds.map(fd => eval_term(prog, fd, stats));
      return Ctr(term.tag, fds);
    }
    case "Rec": {
      stats.app_fun++;
      var arg  = eval_term(prog, term.arg, stats);
      var func = feed(prog, prog, arg, stats);
      if (func.$ !== "Ret") {
        throw new Error("Expected Ret after full application");
      }
      return eval_term(prog, func.trm, stats);
    }
  }
}

// Show
// ----

// Converts a term to a string
function show_term(term: Term): string {
  switch (term.$) {
    case "Var": {
      return term.nam;
    }
    case "Ctr": {
      var fds = term.fds.map(show_term).join(",");
      return `${term.tag}{${fds}}`;
    }
    case "Rec": {
      var arg = show_term(term.arg);
      return `~${arg}`;
    }
  }
}

// Converts a func to a string
function show_func(func: Func, dep: number = 0): string {
  switch (func.$) {
    case "Mat": {
      var zro = show_func(func.zro, dep);
      var one = show_func(func.one, dep);
      return `λ{ 0: ${zro}; 1: ${one} }`;
    }
    case "Lam": {
      var nam = func.nam || String.fromCharCode(97 + dep);
      var bod = show_func(func.bod(Var(nam)), dep + 1);
      return `λ${nam}. ${bod}`;
    }
    case "Ret": {
      return show_term(func.trm);
    }
  }
}

// Formats stats as a report string
function show_stats(stats: Stats): string {
  var total = stats.app_fun + stats.app_lam + stats.app_mat;
  var lines = [
    `Interactions: ${total}`,
    `- APP-FUN: ${stats.app_fun}`,
    `- APP-LAM: ${stats.app_lam}`,
    `- APP-MAT: ${stats.app_mat}`,
  ];
  return lines.join("\n");
}

// Test
// ----

// Decrements 65535 til it hits 0.
// 
// Bits ::=
// | []   ::= 0{}
// | 0,xs ::= 1{0{xs}}
// | 1,xs ::= 1{1{xs}}
// 
// Bool ::=
// | False ::= 0{}
// | True  ::= 1{}
//
// main x        = cond (is0 x) x
// is0 []        = True
// is0 (0 : xs)  = is0 xs
// is0 (1 : xs)  = False
// dec []        = []
// dec (0 : xs)  = 1 : dec xs
// dec (1 : xs)  = 0 : xs
// cond False x  = main (dec x)
// cond True  x  = []

function main() {
  var prog_src = `λ{
    0: λ{
      0: λx. ~1{1{~0{1{x}},x}}
      1: λ{
        0: 1{}
        1: λ{
          0: λxs. ~0{1{xs}}
          1: λxs. 0{}
        }
      }
    }
    1: λ{
      0: λ{
        0: 0{}
        1: λ{
          0: λxs. 1{1{~1{0{xs}}}}
          1: λxs. 1{0{xs}}
        }
      }
      1: λ{
        0: λx. ~0{0{~1{0{x}}}}
        1: λx. 0{}
      }
    }
  }`;

  // Parse the program
  var pp   = { src: prog_src, idx: 0 };
  var prog = parse_func(pp, new Map());
  console.log("prog:", show_func(prog));

  // Build 65535 as 16 one-bits (LSB-first): 1{1{1{1{...1{1{0{}}}...}}}}
  var n = "0{}";
  for (var i = 0; i < 16; i++) {
    n = "1{1{" + n + "}}";
  }
  // Call: ~0{0{<65535>}} = main(65535)
  var input_src = "~0{0{" + n + "}}";

  var tp    = { src: input_src, idx: 0 };
  var input = parse_term(tp, new Map());
  console.log("input:", show_term(input));

  // Evaluate and print
  var stats  = new_stats();
  var result = eval_term(prog, input, stats);
  console.log("result:", show_term(result));
  // Expected: 0{} (halted after counting down to zero)

  // Print interaction stats
  console.log(show_stats(stats));
}

main();
