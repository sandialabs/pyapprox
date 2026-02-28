-- Pandoc Lua filter for PDF book output:
-- 1. Strips HTML landing-page divs from index.qmd
-- 2. Injects \tableofcontents at the start of the document body

local html_classes = {
  ["hero-section"] = true,
  ["section-dark"] = true,
  ["section-light"] = true,
  ["section-bridge"] = true,
  ["section-cta"] = true,
  ["copyright-notice"] = true,
}

function Div(el)
  for _, cls in ipairs(el.classes) do
    if html_classes[cls] then
      return {}
    end
  end
  return el
end

function Pandoc(doc)
  if not FORMAT:match("latex") then
    return doc
  end
  -- Insert \tableofcontents at the very beginning of the body
  local toc = pandoc.RawBlock("latex",
    "\\tableofcontents\n\\newpage")
  table.insert(doc.blocks, 1, toc)
  return doc
end
