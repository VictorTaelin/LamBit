;j// LamBit.ts
// =========
// A minimal functional language based on binary pattern-matching case-trees.
//
//     Func ::=
//       | Lam ::= "λ" Name "." Func
//       | Mat ::= "λ" "{" "0" ":" Func ";" "1" ":" Func ";" "}"
//       | Use ::= "λ" "()" "." Func
//       | Get ::= "λ" "!" Func
//       | Ret ::= Term
//
//     Term ::=
//       | Var ::= Name
//       | Bt0 ::= "0"
//       | Bt1 ::= "1"
//       | Nul ::= "()"
//       | Tup ::= "(" Term "," Term ")"
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
//     F = λ! λ{
//       0: λ! λ{ // computes addition
//         0: λ(). b
//         1: λa. λb. (1,~(0,(a,b)))
//       }
//       1: λ! λ{ // computes multiplication
//         0: λb. 0{}
//         1: λa. λb. ~(0,(b,~(1,(a,b))))
//       }
//     }
//
// Where natural numbers are encoded as:
//
//     Nat ::=
//     | zero ::= (0,())
//     | succ ::= (1,Nat)
//     (aliasing 3n as (1,(1,(1,(0,())))), and so on)
//
// For instance, ~(1,(2n,3n)) evaluates 2n*3n to 6n:
//
//     ~(1,(2n,3n))
//     = ~(0,(3n, ~(1,(1n,3n))))
//     = ~(0,(3n, ~(0,(3n, ~(1,((0,()),3n))))))
//     = ~(0,(3n, ~(0,(3n, (0,())))))
//     = ~(0,(3n, 3n))
//     = (1,~(0,(2n,3n)))
//     = (1,(1,~(0,(1n,3n))))
//     = (1,(1,(1,~(0,((0,()),3n)))))
//     = (1,(1,(1,3n)))
//     = 6n

// Types
// -----

// Function-side: patterns and lambdas
type Get  = { $: "Get", fun: Func };
type Mat  = { $: "Mat", zro: Func, one: Func };
type Lam  = { $: "Lam", nam: string, bod: (x: Term) => Func };
type Use  = { $: "Use", bod: Func };
type Ret  = { $: "Ret", trm: Term };
type Func = Get | Mat | Lam | Use | Ret;

// Data-side: values and expressions
type Var  = { $: "Var", nam: string };
type Bt0  = { $: "Bt0" };
type Bt1  = { $: "Bt1" };
type Nul  = { $: "Nul" };
type Tup  = { $: "Tup", fst: Term, snd: Term };
type Rec  = { $: "Rec", arg: Term };
type Term = Var | Bt0 | Bt1 | Nul | Tup | Rec;

// Interaction stats
type Stats = { app_fun: number, app_lam: number, app_mat: number, app_get: number, app_use: number };

// Constructors
const Get = (fun)      => ({ $: "Get", fun }) as Func;
const Mat = (zro, one) => ({ $: "Mat", zro, one }) as Func;
const Lam = (nam, bod) => ({ $: "Lam", nam, bod }) as Func;
const Use = (bod)      => ({ $: "Use", bod }) as Func;
const Ret = (trm)      => ({ $: "Ret", trm }) as Func;
const Var = (nam)      => ({ $: "Var", nam }) as Term;
const Bt0 = ()         => ({ $: "Bt0" }) as Term;
const Bt1 = ()         => ({ $: "Bt1" }) as Term;
const Nul = ()         => ({ $: "Nul" }) as Term;
const Tup = (fst, snd) => ({ $: "Tup", fst, snd }) as Term;
const Rec = (arg)      => ({ $: "Rec", arg }) as Term;

// Creates a fresh stats counter
function new_stats(): Stats {
  return { app_fun: 0, app_lam: 0, app_mat: 0, app_get: 0, app_use: 0 };
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

// Checks if current position is "()" (null literal)
function is_nul(p: Parser): boolean {
  skip(p);
  return p.src[p.idx] === "(" && p.src[p.idx + 1] === ")";
}

// Checks if current position starts a tuple: "(" but not "()"
function is_tup(p: Parser): boolean {
  skip(p);
  return p.src[p.idx] === "(" && p.src[p.idx + 1] !== ")";
}

// Parses a Func with a name context for HOAS
function parse_func(p: Parser, ctx: Map<string, Term>): Func {
  skip(p);
  // Get: ! Func
  if (match(p, "λ!")) {
    var fun = parse_func(p, ctx);
    return Get(fun);
  }
  // Mat or Lam or Use
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
    // Use: λ(). Func
    if (match(p, "()")) {
      expect(p, ".");
      var bod = parse_func(p, ctx);
      return Use(bod);
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
  // Nul: ()
  if (is_nul(p)) {
    match(p, "()");
    return Nul();
  }
  // Tup: (Term, Term)
  if (is_tup(p)) {
    match(p, "(");
    var fst = parse_term(p, ctx);
    expect(p, ",");
    var snd = parse_term(p, ctx);
    expect(p, ")");
    return Tup(fst, snd);
  }
  // Bt0: 0
  if (peek(p) === "0" && !/[a-zA-Z_0-9{]/.test(p.src[p.idx + 1] || "")) {
    p.idx++;
    return Bt0();
  }
  // Bt1: 1
  if (peek(p) === "1" && !/[a-zA-Z_0-9{]/.test(p.src[p.idx + 1] || "")) {
    p.idx++;
    return Bt1();
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

// Feeds a single term into a Func (handles Get, Mat, Lam, Use)
function feed(prog: Func, func: Func, term: Term, stats: Stats): Func {
  switch (func.$) {
    case "Get": {
      if (term.$ !== "Tup") {
        throw new Error("Get expected a Tup");
      }
      stats.app_get++;
      var result = feed(prog, func.fun, term.fst, stats);
      result = feed(prog, result, term.snd, stats);
      return result;
    }
    case "Lam": {
      stats.app_lam++;
      return func.bod(term);
    }
    case "Mat": {
      if (term.$ !== "Bt0" && term.$ !== "Bt1") {
        throw new Error("Mat expected a Bt0 or Bt1");
      }
      stats.app_mat++;
      return term.$ === "Bt0" ? func.zro : func.one;
    }
    case "Use": {
      if (term.$ !== "Nul") {
        throw new Error("Use expected a Nul");
      }
      stats.app_use++;
      return func.bod;
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
    case "Bt0": {
      return term;
    }
    case "Bt1": {
      return term;
    }
    case "Nul": {
      return term;
    }
    case "Tup": {
      var fst = eval_term(prog, term.fst, stats);
      var snd = eval_term(prog, term.snd, stats);
      return Tup(fst, snd);
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
    case "Bt0": {
      return "0";
    }
    case "Bt1": {
      return "1";
    }
    case "Nul": {
      return "()";
    }
    case "Tup": {
      var fst = show_term(term.fst);
      var snd = show_term(term.snd);
      return `(${fst},${snd})`;
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
    case "Get": {
      var fun = show_func(func.fun, dep);
      return `λ! ${fun}`;
    }
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
    case "Use": {
      var bod = show_func(func.bod, dep);
      return `λ(). ${bod}`;
    }
    case "Ret": {
      return show_term(func.trm);
    }
  }
}

// Formats stats as a report string
function show_stats(stats: Stats): string {
  var total = stats.app_fun + stats.app_lam + stats.app_mat + stats.app_get + stats.app_use;
  var lines = [
    `Interactions: ${total}`,
    `- APP-FUN: ${stats.app_fun}`,
    `- APP-LAM: ${stats.app_lam}`,
    `- APP-MAT: ${stats.app_mat}`,
    `- APP-GET: ${stats.app_get}`,
    `- APP-USE: ${stats.app_use}`,
  ];
  return lines.join("\n");
}

// Test
// ----

// main x          = try (dec x)
// try []       zs = zs
// try (0 : xs) zs = try xs (0 : zs)
// try (1 : xs) zs = main (cat zs (1 : xs))
// cat []       ys = ys
// cat (x : xs) ys = x : (cat xs ys)
// dec []          = []
// dec (0 : xs)    = 1 : dec xs
// dec (1 : xs)    = 0 : xs
function test() {
  var prog_src = `λ! λ{
    0: λ! λ{
      0: λx. ~(0,(1,(~(1,(1,x)),(0,()))))
      1: λ! λ! λ{
        0: λ(). λzs. zs
        1: λ! λ{
          0: λxs. λzs. ~(0,(1,(xs,(1,(0,zs)))))
          1: λxs. λzs. ~(0,(0,~(1,(0,(zs,(1,(1,xs)))))))
        }
      }
    }
    1: λ! λ{
      0: λ! λ! λ{
        0: λ(). λys. ys
        1: λ! λx. λxs. λys. (1,(x,~(1,(0,(xs,ys)))))
      }
      1: λ! λ{
        0: λ(). (0,())
        1: λ! λ{
          0: λxs. (1,(1,~(1,(1,xs))))
          1: λxs. (1,(0,xs))
        }
      }
    }
  }`;

  // Parse the program
  var pp   = { src: prog_src, idx: 0 };
  var prog = parse_func(pp, new Map());
  console.log("prog:", show_func(prog));

  // Build n as 20 one-bits (LSB-first): (1,(1,...(1,(1,(0,()))))) = 2^20 - 1 = 1048575
  var n = "(0,())";
  for (var i = 0; i < 20; i++) {
    n = "(1,(1," + n + "))";
  }
  // Call: ~(0,(0,<n>)) = main(n)
  var input_src = "~(0,(0," + n + "))";
  var input     = parse_term({src: input_src, idx: 0}, new Map());

  // Evaluate and print
  var stats  = new_stats();
  var result = eval_term(prog, input, stats);
  console.log(show_term(result));
  // Expected: (0,()) (halted after counting down to zero)

  // Print interaction stats
  console.log(show_stats(stats));
}

test();
