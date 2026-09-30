-- Phase 4: shares the owner holds after each trade (Form 4 column 5), so scoring can
-- tell a buy that doubles a stake from one that adds 1% to it.
ALTER TABLE trades ADD COLUMN shares_owned_after NUMERIC;
