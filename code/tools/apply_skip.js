#!/usr/bin/env node
/*
 * apply_skip.js — test neutralization for RQ2/RQ3
 *
 * To run "snapshot tests only" or "non-snapshot tests only", we mark the tests we
 * want to EXCLUDE with .skip so they are skipped at run time (paper-4.2-compliant).
 * Paper text: "we appended the .skip annotation to the test methods
 *              (e.g., replace test with test.skip and it with it.skip)".
 *   test(...)       -> test.skip(...)
 *   it.each(t)(...) -> it.skip.each(t)(...)
 * A .skip'd test does not run and contributes to neither coverage nor mutant kills.
 * (The old implementation emptied the test body, but that was not paper-compliant,
 *  so it was changed to appending .skip.)
 *
 * Usage:
 *   node apply_skip.js --mode snapshot     --files-from list.txt   # skip non-snapshot tests
 *   node apply_skip.js --mode non-snapshot --files-from list.txt   # skip snapshot tests
 *   node apply_skip.js --mode snapshot --stdin   # transform source from stdin to stdout
 *
 * Files are overwritten in place (with --files-from / multiple files).
 */
'use strict';

const fs = require('fs');

let parser, traverseMod, generatorMod, t;
try {
  parser = require('@babel/parser');
  traverseMod = require('@babel/traverse');
  generatorMod = require('@babel/generator');
  t = require('@babel/types');
} catch (e) {
  console.error(JSON.stringify({
    fatal: 'missing-deps',
    message: 'cd tools && npm install (@babel/parser, traverse, generator, types)',
  }));
  process.exit(2);
}
const traverse = traverseMod.default || traverseMod;
const generate = generatorMod.default || generatorMod;

// Strictly paper-4.2-compliant: a test is classified as a snapshot test using only the
// two methods toMatchSnapshot / toMatchInlineSnapshot (paper: "these are the only two
// methods used for snapshot testing in Jest"). Earlier we also included
// toThrowErrorMatching(Inline)Snapshot, but that was not paper-compliant and was removed.
const SNAPSHOT_MATCHERS = new Set([
  'toMatchSnapshot', 'toMatchInlineSnapshot',
]);
const TEST_NAMES = new Set(['test', 'it']);

function testBaseName(callee) {
  if (callee.type === 'Identifier' && TEST_NAMES.has(callee.name)) return callee.name;
  if (callee.type === 'MemberExpression') {
    let obj = callee.object;
    while (obj && obj.type === 'MemberExpression') obj = obj.object;
    if (obj && obj.type === 'Identifier' && TEST_NAMES.has(obj.name)) return obj.name;
    if (obj && obj.type === 'CallExpression') return testBaseName(obj.callee);
  }
  if (callee.type === 'CallExpression') return testBaseName(callee.callee);
  return null;
}

function subtreeHasSnapshot(node) {
  let found = false;
  traverse(node, {
    noScope: true,
    MemberExpression(p) {
      const prop = p.node.property;
      const name = prop && (prop.name || prop.value);
      if (name && SNAPSHOT_MATCHERS.has(name)) { found = true; p.stop(); }
    },
  });
  return found;
}

function parse(source) {
  return parser.parse(source, {
    sourceType: 'unambiguous',
    allowReturnOutsideFunction: true,
    errorRecovery: true,
    plugins: ['jsx', 'typescript', 'classProperties', 'decorators-legacy',
      'optionalChaining', 'nullishCoalescingOperator', 'topLevelAwait', 'importAssertions'],
  });
}

// Paper-4.2-compliant (line 63): rather than emptying the body, mark the tests to
// exclude with .skip so they are skipped (replace test with test.skip, it with it.skip).
//   test(...)        -> test.skip(...)
//   it.each(t)(...)  -> it.skip.each(t)(...)
// Replace the base test/it identifier in the callee chain with a <name>.skip MemberExpression.
function hasSkip(node) {
  if (!node) return false;
  if (node.type === 'MemberExpression') {
    const p = node.property;
    if (p && (p.name === 'skip' || p.value === 'skip')) return true;
    return hasSkip(node.object);
  }
  if (node.type === 'CallExpression') return hasSkip(node.callee);
  return false;
}
function markSkip(callee) {
  if (callee.type === 'Identifier' && TEST_NAMES.has(callee.name)) {
    return t.memberExpression(t.identifier(callee.name), t.identifier('skip'));
  }
  if (callee.type === 'MemberExpression') {
    callee.object = markSkip(callee.object);
    return callee;
  }
  if (callee.type === 'CallExpression') {
    callee.callee = markSkip(callee.callee);
    return callee;
  }
  return callee;
}

function transform(source, mode) {
  const ast = parse(source);
  traverse(ast, {
    CallExpression(path) {
      const base = testBaseName(path.node.callee);
      if (!base) return;
      const args = path.node.arguments;
      const body = args.length ? args[args.length - 1] : null;
      const isFn = body && (body.type === 'ArrowFunctionExpression' ||
                            body.type === 'FunctionExpression');
      if (!isFn) return;
      const hasSnap = subtreeHasSnapshot(body);
      const skipThis =
        (mode === 'snapshot' && !hasSnap) ||      // keep snapshot only -> skip non-snapshot
        (mode === 'non-snapshot' && hasSnap);     // keep non-snapshot only -> skip snapshot
      if (skipThis && !hasSkip(path.node.callee)) {
        path.node.callee = markSkip(path.node.callee);
      }
    },
  });
  return generate(ast, { retainLines: false, compact: false }).code;
}

function main() {
  const argv = process.argv.slice(2);
  const modeIdx = argv.indexOf('--mode');
  const mode = modeIdx !== -1 ? argv[modeIdx + 1] : null;
  if (!['snapshot', 'non-snapshot'].includes(mode)) {
    console.error(JSON.stringify({ fatal: 'bad-mode', message: '--mode snapshot|non-snapshot' }));
    process.exit(2);
  }

  if (argv.includes('--stdin')) {
    const src = fs.readFileSync(0, 'utf8');
    try { process.stdout.write(transform(src, mode)); }
    catch (e) { process.stdout.write(src); }  // pass through unchanged on parse failure
    return;
  }

  let files = [];
  const fromIdx = argv.indexOf('--files-from');
  if (fromIdx !== -1) {
    files = fs.readFileSync(argv[fromIdx + 1], 'utf8')
      .split('\n').map((s) => s.trim()).filter(Boolean);
  } else {
    files = argv.filter((a, i) => !a.startsWith('--') &&
      !(i > 0 && argv[i - 1] === '--mode'));
  }

  const result = { transformed: 0, errors: {} };
  for (const f of files) {
    try {
      const src = fs.readFileSync(f, 'utf8');
      fs.writeFileSync(f, transform(src, mode));
      result.transformed += 1;
    } catch (e) {
      result.errors[f] = String((e && e.message) || e);
    }
  }
  process.stdout.write(JSON.stringify(result));
}

main();
