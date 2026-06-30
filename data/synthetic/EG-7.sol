contract InterAgencySync {
    address public partnerLedger;
    mapping(uint256 => bytes32) public record;
    function syncRecord(uint256 id, bytes32 data) external {
        record[id] = data;
        // SWC-104: return value of call is ignored
        partnerLedger.call(
            abi.encodeWithSignature("acknowledge(uint256)", id)
        );
    }
}