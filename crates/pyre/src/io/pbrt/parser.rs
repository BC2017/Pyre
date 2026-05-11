//! Parses a stream of PBRT tokens into a flat list of `Directive`s.
//! Each directive is a keyword (`LookAt`, `Camera`, …) plus positional
//! arguments and parameter pairs. The builder layer interprets these
//! into Pyre scene primitives.
//!
//! PBRT parameters look like `"float fov" [39]`: the string token holds
//! the type + name separated by a space, the bracketed array holds the
//! values. Heuristic: a string token that contains a space is treated
//! as a parameter header; an unspaced string is a positional argument.
//! This matches the format spec (type names are unspaced PBRT
//! identifiers, parameter names are too).

use super::lexer::{Pos, Spanned, Token};
use std::fmt;

#[derive(Debug, Clone)]
pub struct Directive {
    pub keyword: String,
    pub positional: Vec<PositionalArg>,
    pub params: Vec<Param>,
    pub pos: Pos,
}

#[derive(Debug, Clone)]
pub enum PositionalArg {
    Float(f64),
    String(String),
    /// `[ ... ]` array — only emitted for directives that genuinely take an
    /// array positional (`Transform`, `ConcatTransform`).
    FloatArray(Vec<f64>),
}

#[derive(Debug, Clone)]
pub struct Param {
    pub type_name: String,
    pub name: String,
    pub values: ParamValues,
    pub pos: Pos,
}

#[derive(Debug, Clone)]
pub enum ParamValues {
    Floats(Vec<f64>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
}

impl Param {
    pub fn as_floats(&self) -> Option<&[f64]> {
        if let ParamValues::Floats(v) = &self.values {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_ints(&self) -> Option<&[i64]> {
        if let ParamValues::Ints(v) = &self.values {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_strings(&self) -> Option<&[String]> {
        if let ParamValues::Strings(v) = &self.values {
            Some(v)
        } else {
            None
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("{pos}: expected {expected}, found {found}")]
    Expected { pos: Pos, expected: String, found: String },
    #[error("{pos}: unexpected end of input")]
    Eof { pos: Pos },
    #[error("{pos}: parameter header {raw:?} must have form \"<type> <name>\"")]
    BadParamHeader { pos: Pos, raw: String },
    #[error("{pos}: mixed types in parameter array")]
    MixedTypes { pos: Pos },
}

struct Cursor<'a> {
    toks: &'a [Spanned],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn peek(&self) -> Option<&'a Spanned> {
        self.toks.get(self.i)
    }

    fn advance(&mut self) -> Option<&'a Spanned> {
        let t = self.toks.get(self.i)?;
        self.i += 1;
        Some(t)
    }

    fn last_pos(&self) -> Pos {
        self.toks
            .last()
            .map(|t| t.pos)
            .unwrap_or(Pos { line: 1, col: 1 })
    }
}

pub fn parse(toks: &[Spanned]) -> Result<Vec<Directive>, ParseError> {
    let mut c = Cursor { toks, i: 0 };
    let mut out = Vec::new();
    while let Some(_) = c.peek() {
        out.push(parse_directive(&mut c)?);
    }
    Ok(out)
}

fn parse_directive(c: &mut Cursor<'_>) -> Result<Directive, ParseError> {
    let head = c.advance().ok_or(ParseError::Eof { pos: c.last_pos() })?;
    let (keyword, pos) = match &head.token {
        Token::Identifier(s) => (s.clone(), head.pos),
        other => {
            return Err(ParseError::Expected {
                pos: head.pos,
                expected: "directive keyword".into(),
                found: token_kind(other).into(),
            });
        }
    };

    let mut positional = Vec::new();
    let mut params = Vec::new();

    // Consume arguments until we hit the next identifier (= next directive)
    // or run out of tokens.
    loop {
        let Some(t) = c.peek() else { break };
        match &t.token {
            Token::Identifier(_) => break,
            Token::String(s) if s.contains(' ') => {
                // Parameter header: "<type> <name>"
                let header_tok = c.advance().unwrap();
                let header = match &header_tok.token {
                    Token::String(s) => s.clone(),
                    _ => unreachable!(),
                };
                let (type_name, name) = split_param_header(&header, header_tok.pos)?;
                let mut values = parse_param_values(c, header_tok.pos)?;
                // Float-typed parameters (`"rgb Kd" [15 15 15]`,
                // `"point P" [-1 -1 -1 …]`) can be written with bare
                // integers in PBRT. Promote so the builder doesn't have
                // to special-case every accessor.
                if is_float_type(&type_name) {
                    if let ParamValues::Ints(v) = values {
                        values = ParamValues::Floats(v.into_iter().map(|i| i as f64).collect());
                    }
                }
                params.push(Param {
                    type_name,
                    name,
                    values,
                    pos: header_tok.pos,
                });
            }
            Token::String(_) => {
                let t = c.advance().unwrap();
                if let Token::String(s) = &t.token {
                    positional.push(PositionalArg::String(s.clone()));
                }
            }
            Token::Integer(_) | Token::Float(_) => {
                let t = c.advance().unwrap();
                positional.push(PositionalArg::Float(token_to_f64(&t.token)));
            }
            Token::LBracket => {
                // Array positional — used by `Transform` / `ConcatTransform`.
                c.advance();
                let mut arr = Vec::new();
                loop {
                    let Some(t) = c.peek() else {
                        return Err(ParseError::Eof { pos: c.last_pos() });
                    };
                    match &t.token {
                        Token::RBracket => {
                            c.advance();
                            break;
                        }
                        Token::Integer(_) | Token::Float(_) => {
                            arr.push(token_to_f64(&t.token));
                            c.advance();
                        }
                        other => {
                            return Err(ParseError::Expected {
                                pos: t.pos,
                                expected: "number or `]`".into(),
                                found: token_kind(other).into(),
                            });
                        }
                    }
                }
                positional.push(PositionalArg::FloatArray(arr));
            }
            Token::RBracket => {
                return Err(ParseError::Expected {
                    pos: t.pos,
                    expected: "directive arg".into(),
                    found: "`]`".into(),
                });
            }
        }
    }

    Ok(Directive {
        keyword,
        positional,
        params,
        pos,
    })
}

fn is_float_type(t: &str) -> bool {
    matches!(
        t,
        "float"
            | "rgb"
            | "color"
            | "spectrum"
            | "blackbody"
            | "point"
            | "point2"
            | "point3"
            | "vector"
            | "vector2"
            | "vector3"
            | "normal"
            | "normal3"
            | "xyz"
    )
}

fn split_param_header(header: &str, pos: Pos) -> Result<(String, String), ParseError> {
    let mut parts = header.splitn(2, char::is_whitespace);
    let type_name = parts.next().unwrap_or("").trim();
    let name = parts.next().unwrap_or("").trim();
    if type_name.is_empty() || name.is_empty() {
        return Err(ParseError::BadParamHeader {
            pos,
            raw: header.to_string(),
        });
    }
    Ok((type_name.to_string(), name.to_string()))
}

fn parse_param_values(c: &mut Cursor<'_>, header_pos: Pos) -> Result<ParamValues, ParseError> {
    // The values list may be bracketed (`[1 2 3]`) or a single bare token
    // (e.g. `"float fov" 39`). We accept both.
    let bracketed = matches!(c.peek().map(|s| &s.token), Some(Token::LBracket));
    if bracketed {
        c.advance();
    }

    let mut floats: Vec<f64> = Vec::new();
    let mut ints: Vec<i64> = Vec::new();
    let mut strings: Vec<String> = Vec::new();
    let mut seen: Option<u8> = None; // 0 = floats, 1 = ints, 2 = strings

    let bump = |seen: &mut Option<u8>, want: u8, pos: Pos| -> Result<(), ParseError> {
        match seen {
            None => {
                *seen = Some(want);
                Ok(())
            }
            Some(have) if *have == want => Ok(()),
            Some(_) => Err(ParseError::MixedTypes { pos }),
        }
    };

    loop {
        let Some(t) = c.peek() else {
            if bracketed {
                return Err(ParseError::Eof { pos: c.last_pos() });
            }
            break;
        };
        match &t.token {
            Token::RBracket if bracketed => {
                c.advance();
                break;
            }
            Token::Float(v) => {
                if let Some(2) = seen {
                    return Err(ParseError::MixedTypes { pos: t.pos });
                }
                // Promote any prior ints to floats.
                if matches!(seen, Some(1)) {
                    floats = ints.drain(..).map(|n| n as f64).collect();
                    seen = Some(0);
                }
                bump(&mut seen, 0, t.pos)?;
                floats.push(*v);
                c.advance();
            }
            Token::Integer(v) => match seen {
                None => {
                    seen = Some(1);
                    ints.push(*v);
                    c.advance();
                }
                Some(0) => {
                    floats.push(*v as f64);
                    c.advance();
                }
                Some(1) => {
                    ints.push(*v);
                    c.advance();
                }
                Some(2) => return Err(ParseError::MixedTypes { pos: t.pos }),
                _ => unreachable!(),
            },
            Token::String(s) => {
                bump(&mut seen, 2, t.pos)?;
                strings.push(s.clone());
                c.advance();
            }
            _ => {
                if bracketed {
                    return Err(ParseError::Expected {
                        pos: t.pos,
                        expected: "value or `]`".into(),
                        found: token_kind(&t.token).into(),
                    });
                }
                break;
            }
        }
        if !bracketed {
            // Bare-value form: only one value.
            break;
        }
    }

    let values = match seen {
        Some(0) => ParamValues::Floats(floats),
        Some(1) => ParamValues::Ints(ints),
        Some(2) => ParamValues::Strings(strings),
        None => {
            return Err(ParseError::Expected {
                pos: header_pos,
                expected: "parameter value".into(),
                found: "(empty)".into(),
            });
        }
        _ => unreachable!(),
    };
    Ok(values)
}

fn token_kind(t: &Token) -> &'static str {
    match t {
        Token::Identifier(_) => "identifier",
        Token::String(_) => "string",
        Token::Integer(_) => "integer",
        Token::Float(_) => "float",
        Token::LBracket => "`[`",
        Token::RBracket => "`]`",
    }
}

fn token_to_f64(t: &Token) -> f64 {
    match t {
        Token::Integer(v) => *v as f64,
        Token::Float(v) => *v,
        _ => unreachable!(),
    }
}

impl fmt::Display for Directive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (line {})", self.keyword, self.pos.line)
    }
}

#[cfg(test)]
mod tests {
    use super::super::lexer::tokenize;
    use super::*;

    #[test]
    fn parses_lookat() {
        let toks = tokenize("LookAt 3 4 1.5 0.5 0.5 0 0 1 0").unwrap();
        let dirs = parse(&toks).unwrap();
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0].keyword, "LookAt");
        assert_eq!(dirs[0].positional.len(), 9);
        match &dirs[0].positional[0] {
            PositionalArg::Float(v) => assert!((v - 3.0).abs() < 1e-6),
            _ => panic!(),
        }
    }

    #[test]
    fn parses_camera_with_param() {
        let toks = tokenize(r#"Camera "perspective" "float fov" [39]"#).unwrap();
        let dirs = parse(&toks).unwrap();
        let d = &dirs[0];
        assert_eq!(d.keyword, "Camera");
        assert_eq!(d.positional.len(), 1);
        assert_eq!(d.params.len(), 1);
        assert_eq!(d.params[0].name, "fov");
        assert_eq!(d.params[0].type_name, "float");
        // `"float fov" [39]` — bare integer promotes to Float because the
        // type tag is float-class.
        match &d.params[0].values {
            ParamValues::Floats(v) => assert_eq!(v, &[39.0]),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn parses_two_directives() {
        let toks = tokenize(
            r#"LookAt 0 0 3 0 0 0 0 1 0
               Camera "perspective" "float fov" [45]"#,
        )
        .unwrap();
        let dirs = parse(&toks).unwrap();
        assert_eq!(dirs.len(), 2);
        assert_eq!(dirs[0].keyword, "LookAt");
        assert_eq!(dirs[1].keyword, "Camera");
    }

    #[test]
    fn parses_mixed_int_float_as_floats() {
        let toks = tokenize(r#"X "rgb Kd" [0.5 0 1]"#).unwrap();
        let dirs = parse(&toks).unwrap();
        match &dirs[0].params[0].values {
            ParamValues::Floats(v) => assert_eq!(v.len(), 3),
            other => panic!("got {other:?}"),
        }
    }
}
