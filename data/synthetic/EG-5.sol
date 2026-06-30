contract CredentialRevocation {
  address public registrar;
  mapping(bytes32 => bool) public revoked;
  constructor() { registrar = msg.sender; }
  function revoke(bytes32 credHash) external {
    require(tx.origin == registrar, "auth"); // SWC-115
    revoked[credHash] = true;
  }
}