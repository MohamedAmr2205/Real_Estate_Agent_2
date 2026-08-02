# Meridian Realty — Fair Housing Advertising Policy

This document is exposed via `resources/read` (URI: `policy://fair-housing`)
so the model can READ it before drafting any listing description, rather
than the rule being buried inside a tool's code where it can't be inspected
or updated independently.

## Rules for listing descriptions

1. Do not reference or imply the race, color, religion, sex, national
   origin, familial status, or disability of current or prospective
   residents/buyers.
2. Do not use phrases that signal a preference for or against a protected
   class, e.g. "perfect for young professionals," "no children," "ideal
   for a Christian family," "walking distance to [religious institution]"
   used as a selling point tied to a group.
3. Describe the PROPERTY (square footage, layout, finishes, location,
   amenities) — never describe or speculate about who should live there.
4. Accessibility features (ramps, elevators, wide doorways) may be listed
   as factual property features, not as a signal about who the property
   is "for."
5. Neighborhood descriptions must stay factual (distance to transit,
   schools, parks) and avoid subjective character judgments about the
   people who live there.

## How this is used

`draft_listing_description` (see `prompts/`) instructs the model to read
this resource before generating any description text, and to flag any
seed language provided by an agent that appears to violate these rules
instead of silently rewriting it.